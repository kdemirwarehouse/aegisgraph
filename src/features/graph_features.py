"""NetworkX tabanlı, sızıntısız (point-in-time) graf özellikleri.

İki graf artımlı olarak tutulur:
  * `flow`  (nx.DiGraph)  hesap → hesap para akışı; kenar özellikleri: count, amount, last_ts
  * `links` (nx.Graph)    bipartit hesap ↔ cihaz / IP-/24 bloğu ilişkisi

Protokol (her işlem için):  `features(...)`  →  `update(...)`
`features` yalnızca o ana kadar `update` edilmiş (yani geçmiş) işlemleri görür. Böylece batch eğitim
ve online skorlama aynı kod yolundan geçer ve gelecekten bilgi sızamaz.

Pahalı global metrikler (PageRank, clustering) her takvim gününün ilk işleminde, o güne kadarki
grafın anlık görüntüsünden yeniden hesaplanır (gün içi değerler bir önceki günün sonunu yansıtır).
"""

from __future__ import annotations

from collections import Counter, deque

import networkx as nx

DAY_SECONDS = 86_400

GRAPH_FEATURES: tuple[str, ...] = (
    "sender_in_degree",
    "sender_out_degree",
    "receiver_in_degree",
    "receiver_out_degree",
    "sender_in_out_ratio",
    "receiver_in_out_ratio",
    "sender_pagerank",
    "receiver_pagerank",
    "sender_clustering",
    "receiver_clustering",
    "shared_device_account_count",
    "shared_ip_account_count",
    "receiver_unique_senders_24h",
    "has_reciprocal_flow",
    "short_cycle_risk",
)


def ip_block(ip: str | None) -> str | None:
    """IPv4 adresinin /24 bloğu ('85.105.214.41' → '85.105.214')."""
    if not ip:
        return None
    return ip.rsplit(".", 1)[0]


class GraphFeatureState:
    def __init__(
        self,
        *,
        merchant_accounts: set[str],
        flow_window_seconds: int = DAY_SECONDS,
        max_cycle_hops: int = 7,
    ) -> None:
        self.flow = nx.DiGraph()
        self.links = nx.Graph()
        self.window = flow_window_seconds
        self.max_hops = max_cycle_hops
        self._merchants = merchant_accounts
        self._pagerank: dict[str, float] = {}
        self._clustering: dict[str, float] = {}
        self._snapshot_day: int | None = None
        self._recv_window: dict[str, deque[tuple[float, str]]] = {}
        self._recv_counter: dict[str, Counter[str]] = {}

    # ------------------------------------------------------------------ özellik
    def features(
        self, ts: float, sender: str, receiver: str, device: str | None, ip: str | None
    ) -> dict[str, float]:
        self._maybe_refresh_snapshot(ts)
        s_in, s_out = self._degrees(sender)
        r_in, r_out = self._degrees(receiver)
        hops = self._return_path_hops(sender, receiver, ts)
        return {
            "sender_in_degree": s_in,
            "sender_out_degree": s_out,
            "receiver_in_degree": r_in,
            "receiver_out_degree": r_out,
            "sender_in_out_ratio": (s_in + 1) / (s_out + 1),
            "receiver_in_out_ratio": (r_in + 1) / (r_out + 1),
            "sender_pagerank": self._pagerank.get(sender, 0.0),
            "receiver_pagerank": self._pagerank.get(receiver, 0.0),
            "sender_clustering": self._clustering.get(sender, 0.0),
            "receiver_clustering": self._clustering.get(receiver, 0.0),
            "shared_device_account_count": self._shared_accounts(("D", device), sender) if device else 0,
            "shared_ip_account_count": self._shared_accounts(("I", ip_block(ip)), sender) if ip else 0,
            "receiver_unique_senders_24h": self._unique_recent_senders(receiver, sender, ts),
            "has_reciprocal_flow": float(hops == 1),
            # Döngü uzunluğu L = hops + 1 → risk = 2 / L (karşılıklı akış = 1.0, 3'lü döngü = 0.67 ...)
            "short_cycle_risk": 2.0 / (hops + 1) if hops else 0.0,
        }

    # ---------------------------------------------------------------- güncelle
    def update(
        self, ts: float, sender: str, receiver: str, device: str | None, ip: str | None, amount: float
    ) -> None:
        if self.flow.has_edge(sender, receiver):
            edge = self.flow[sender][receiver]
            edge["count"] += 1
            edge["amount"] += amount
            edge["last_ts"] = ts
        else:
            self.flow.add_edge(sender, receiver, count=1, amount=amount, last_ts=ts)

        account_node = ("A", sender)
        if device:
            self.links.add_edge(account_node, ("D", device))
        if ip:
            self.links.add_edge(account_node, ("I", ip_block(ip)))

        window = self._recv_window.setdefault(receiver, deque())
        counter = self._recv_counter.setdefault(receiver, Counter())
        window.append((ts, sender))
        counter[sender] += 1

    # ---------------------------------------------------------------- iç işler
    def _degrees(self, account: str) -> tuple[int, int]:
        if account not in self.flow:
            return 0, 0
        return self.flow.in_degree(account), self.flow.out_degree(account)

    def _shared_accounts(self, node: tuple[str, str | None], sender: str) -> int:
        """Bu cihazı / IP bloğunu kullanan farklı hesap sayısı (mevcut gönderici dahil)."""
        if node not in self.links:
            return 1
        neighbours = self.links.adj[node]
        return len(neighbours) + (0 if ("A", sender) in neighbours else 1)

    def _unique_recent_senders(self, receiver: str, sender: str, ts: float) -> int:
        """Alıcıya son pencerede para gönderen farklı hesap sayısı (mevcut gönderici dahil)."""
        window = self._recv_window.get(receiver)
        if window is None:
            return 1
        counter = self._recv_counter[receiver]
        cutoff = ts - self.window
        while window and window[0][0] < cutoff:
            _, old = window.popleft()
            counter[old] -= 1
            if counter[old] == 0:
                del counter[old]
        return len(counter) + (0 if sender in counter else 1)

    def _return_path_hops(self, sender: str, receiver: str, ts: float) -> int | None:
        """Pencere içindeki kenarlarla alıcı → … → gönderici en kısa yolun adım sayısı (BFS)."""
        if receiver in self._merchants or receiver not in self.flow or sender not in self.flow:
            return None
        cutoff = ts - self.window
        succ = self.flow.succ
        frontier, seen = [receiver], {receiver}
        for depth in range(1, self.max_hops + 1):
            nxt: list[str] = []
            for node in frontier:
                for nbr, data in succ[node].items():
                    if data["last_ts"] < cutoff or nbr in seen:
                        continue
                    if nbr == sender:
                        return depth
                    seen.add(nbr)
                    if nbr not in self._merchants:  # üye işyerleri para göndermez → çıkmaz sokak
                        nxt.append(nbr)
            if not nxt:
                return None
            frontier = nxt
        return None

    def _maybe_refresh_snapshot(self, ts: float) -> None:
        day = int(ts // DAY_SECONDS)
        if day == self._snapshot_day:
            return
        self._snapshot_day = day
        n = self.flow.number_of_nodes()
        if n == 0:
            return
        # Ölçekten bağımsız olsun diye PageRank × N (ortalama düğüm = 1.0)
        pr = nx.pagerank(self.flow, weight="count")
        self._pagerank = {node: value * n for node, value in pr.items()}
        # Clustering: üye işyerleri hariç P2P alt grafında (halka/topluluk yapısı)
        p2p = nx.Graph(self.flow.subgraph(node for node in self.flow if node not in self._merchants))
        self._clustering = nx.clustering(p2p)
