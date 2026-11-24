"""Otonom finansal soruşturma ajanı için salt-okunur ve belirlenimci kanıt toplama araçları.

Tüm araçlar:
  * Veritabanına hiçbir yazma yapmaz (read-only).
  * `GROUND_TRUTH_COLUMNS` etiketlerini asla okumaz ve çıktıya dahil etmez.
  * Tüm IBAN bilgilerini `mask_iban` (`TR** **** **** **** **** **12 34`) ile maskeler; hesap sahibi
    adı KVKK gereği `mask_holder_name` ile maskelenir ve her çıktı son olarak `redact_ibans`
    süzgecinden geçer (ham IBAN hiçbir koşulda dönmez).
  * Eşikler ve liste sınırları yalnızca `settings.agent` üzerinden okunur.
  * Her bulguyu kaynak işlem/cihaz/IP bloğu kimlikleriyle kanıtlar.
"""

from __future__ import annotations

import itertools
import logging
from datetime import datetime, timedelta
from statistics import median
from typing import Any

import networkx as nx
from sqlalchemy import Engine, func, or_, select

from config.settings import Settings, get_settings
from src.compat import UTC
from src.database.db import get_engine, session_scope
from src.database.models import (
    Account,
    AccountDevice,
    Alert,
    Device,
    Transaction,
    mask_holder_name,
    mask_iban,
    redact_ibans,
)
from src.features.graph_features import ip_block
from src.features.pipeline import FeatureExtractor

logger = logging.getLogger(__name__)

_DAY_SECONDS = 86_400


def _resolve_ref_time(
    session: Any,
    account_id: str,
    reference_time: datetime | None,
) -> datetime | None:
    if reference_time is not None:
        return reference_time
    return session.scalar(
        select(func.max(Transaction.timestamp)).where(
            or_(
                Transaction.sender_account_id == account_id,
                Transaction.receiver_account_id == account_id,
            )
        )
    )


def get_account_profile(
    account_id: str,
    *,
    engine: Engine | None = None,
    reference_time: datetime | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Hesap yaşı, segment, KYC seviyesi, ortalama bakiye göstergesi ve geçmiş alarm sayısını döndürür.

    `reference_time` verilirse yalnızca o andan önceki işlem ve alarmlar dikkate alınır (point-in-time).
    """
    cfg = settings or get_settings()
    db_engine = engine or get_engine()
    with session_scope(db_engine) as session:
        acc = session.get(Account, account_id)
        if acc is None:
            raise KeyError(f"Hesap bulunamadı: {account_id}")

        ref_ts = _resolve_ref_time(session, account_id, reference_time) or acc.opened_at
        age_days = max(0.0, (ref_ts - acc.opened_at).total_seconds() / _DAY_SECONDS)

        sent_filters = [Transaction.sender_account_id == account_id]
        recv_filters = [Transaction.receiver_account_id == account_id]
        if reference_time is not None:
            sent_filters.append(Transaction.timestamp <= reference_time)
            recv_filters.append(Transaction.timestamp <= reference_time)

        sent_count, sent_sum_minor = session.execute(
            select(func.count(Transaction.transaction_id), func.coalesce(func.sum(Transaction.amount_minor), 0)).where(
                *sent_filters
            )
        ).one()
        recv_count, recv_sum_minor = session.execute(
            select(func.count(Transaction.transaction_id), func.coalesce(func.sum(Transaction.amount_minor), 0)).where(
                *recv_filters
            )
        ).one()

        sent_total_try = float(sent_sum_minor or 0) / 100.0
        recv_total_try = float(recv_sum_minor or 0) / 100.0
        sent_n = int(sent_count or 0)
        recv_n = int(recv_count or 0)
        avg_tx_try = sent_total_try / sent_n if sent_n else 0.0
        # Mevcut şemada ayrı bakiye kolonu olmadığı için gelen-giden akışından tutarlı gösterge türetilir
        net_flow_try = recv_total_try - sent_total_try
        avg_balance_try = max(0.0, avg_tx_try * 2.5 + max(net_flow_try, 0.0) * 0.25)

        device_ids = [
            str(dev_id)
            for dev_id in session.scalars(
                select(AccountDevice.device_id)
                .where(AccountDevice.account_id == account_id)
                .order_by(AccountDevice.last_seen.desc())
            ).all()
        ]

        alert_stmt = (
            select(Alert.alert_id, Alert.risk_score, Alert.status)
            .join(Transaction, Alert.transaction_id == Transaction.transaction_id)
            .where(
                or_(
                    Transaction.sender_account_id == account_id,
                    Transaction.receiver_account_id == account_id,
                )
            )
            .order_by(Alert.alert_id.desc())
        )
        if reference_time is not None:
            alert_stmt = alert_stmt.where(Transaction.timestamp <= reference_time)
        alert_rows = session.execute(alert_stmt).all()

        alert_ids = [int(row[0]) for row in alert_rows]
        max_alert_score = max((float(row[1]) for row in alert_rows), default=0.0)
        open_alerts = sum(1 for row in alert_rows if row[2].value in {"OPEN", "INVESTIGATING", "ESCALATED"})

        id_cap = cfg.agent.max_evidence_ids
        return redact_ibans({
            "account_id": acc.account_id,
            "holder_name_masked": mask_holder_name(acc.holder_name),
            "masked_iban": mask_iban(acc.iban),
            "segment": acc.account_type.value,
            "account_type": acc.account_type.value,
            "kyc_level": int(acc.kyc_level),
            "city": acc.city,
            "merchant_category": acc.merchant_category.value if acc.merchant_category else None,
            "opened_at": acc.opened_at.isoformat(),
            "reference_time": ref_ts.isoformat(),
            "account_age_days": round(age_days, 2),
            "sent_tx_count": sent_n,
            "received_tx_count": recv_n,
            "sent_total_amount": round(sent_total_try, 2),
            "received_total_amount": round(recv_total_try, 2),
            "average_tx_amount": round(avg_tx_try, 2),
            "average_balance": round(avg_balance_try, 2),
            "registered_device_count": len(device_ids),
            "registered_device_ids": device_ids[:id_cap],
            "past_alert_count": len(alert_ids),
            "open_alert_count": open_alerts,
            "max_alert_score": round(max_alert_score, 4),
            "alert_ids": alert_ids[:id_cap],
        })


def get_recent_transactions(
    account_id: str,
    limit: int | None = None,
    *,
    engine: Engine | None = None,
    reference_time: datetime | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Müşterinin son işlemlerini, kanal/konum değişimlerini ve tutar sapmalarını döndürür."""
    cfg = settings or get_settings()
    resolved_limit = limit or cfg.agent.recent_tx_limit
    db_engine = engine or get_engine()

    with session_scope(db_engine) as session:
        if session.get(Account, account_id) is None:
            raise KeyError(f"Hesap bulunamadı: {account_id}")

        ref_ts = _resolve_ref_time(session, account_id, reference_time)
        stmt = (
            select(
                Transaction.transaction_id,
                Transaction.timestamp,
                Transaction.sender_account_id,
                Transaction.receiver_account_id,
                Transaction.device_id,
                Transaction.amount_minor,
                Transaction.currency,
                Transaction.channel,
                Transaction.transaction_type,
                Transaction.ip_address,
                Transaction.geo_city,
                Transaction.geo_country,
            )
            .where(
                or_(
                    Transaction.sender_account_id == account_id,
                    Transaction.receiver_account_id == account_id,
                )
            )
            .order_by(Transaction.timestamp.asc(), Transaction.transaction_id.asc())
        )
        if ref_ts is not None:
            stmt = stmt.where(Transaction.timestamp <= ref_ts)
        all_rows = session.execute(stmt).all()

        recent_counterparties = {
            row.receiver_account_id if row.sender_account_id == account_id else row.sender_account_id
            for row in all_rows[-resolved_limit:]
        }
        counterparty_ibans = {
            str(acc_id): mask_iban(iban)
            for acc_id, iban in session.execute(
                select(Account.account_id, Account.iban).where(Account.account_id.in_(recent_counterparties))
            ).all()
        }

    sent_amounts = [float(row.amount_minor) / 100.0 for row in all_rows if row.sender_account_id == account_id]
    avg_amount = sum(sent_amounts) / len(sent_amounts) if sent_amounts else 0.0
    med_amount = float(median(sent_amounts)) if sent_amounts else 0.0
    prior_median = float(median(sent_amounts[:-1])) if len(sent_amounts) >= 2 else med_amount

    recent_rows = list(reversed(all_rows[-resolved_limit:]))
    chronological_recent = list(reversed(recent_rows))

    channel_switches = 0
    location_changes = 0
    foreign_countries: set[str] = set()
    foreign_tx_count = 0
    night_tx_count = 0
    micro_tx_times: list[datetime] = []
    high_value_count = 0
    max_deviation_ratio = 0.0

    prev_channel: str | None = None
    prev_location: tuple[str | None, str] | None = None
    home_country = cfg.features.home_country
    night_start = cfg.features.night_start_hour
    night_end = cfg.features.night_end_hour
    high_val_thr = cfg.risk.high_value_amount_try
    micro_amount = cfg.agent.micro_tx_amount_try
    burst_window = cfg.agent.micro_burst_window_seconds

    for row in chronological_recent:
        amount_try = float(row.amount_minor) / 100.0
        ch = row.channel.value
        loc = (row.geo_city, row.geo_country)

        if row.sender_account_id == account_id:
            if prev_channel is not None and ch != prev_channel:
                channel_switches += 1
            prev_channel = ch

            if prev_location is not None and loc != prev_location:
                location_changes += 1
            prev_location = loc

            if prior_median > 0:
                max_deviation_ratio = max(max_deviation_ratio, amount_try / prior_median)
            if amount_try >= high_val_thr:
                high_value_count += 1
            if amount_try <= micro_amount:
                micro_tx_times.append(row.timestamp)

        if row.geo_country != home_country:
            foreign_tx_count += 1
            foreign_countries.add(row.geo_country)
        if night_start <= row.timestamp.hour < night_end:
            night_tx_count += 1

    # `micro_burst_window_seconds` penceresindeki maksimum mikro işlem kümesi (Card Testing sinyali)
    micro_burst_count = 0
    for idx, t0 in enumerate(micro_tx_times):
        burst = sum(1 for t1 in micro_tx_times[idx:] if (t1 - t0).total_seconds() <= burst_window)
        micro_burst_count = max(micro_burst_count, burst)

    serialized_txs = []
    for row in recent_rows:
        amount_try = float(row.amount_minor) / 100.0
        is_out = row.sender_account_id == account_id
        counterparty = row.receiver_account_id if is_out else row.sender_account_id
        serialized_txs.append({
            "transaction_id": row.transaction_id,
            "timestamp": row.timestamp.isoformat(),
            "direction": "OUT" if is_out else "IN",
            "counterparty_account_id": counterparty,
            "counterparty_masked_iban": counterparty_ibans.get(str(counterparty), mask_iban(None)),
            "amount_minor": int(row.amount_minor),
            "amount": round(amount_try, 2),
            "currency": row.currency,
            "channel": row.channel.value,
            "transaction_type": row.transaction_type.value,
            "device_id": row.device_id,
            "ip_address": row.ip_address,
            "geo_city": row.geo_city,
            "geo_country": row.geo_country,
            "amount_to_median_ratio": round(amount_try / prior_median, 2) if prior_median > 0 else None,
        })

    return redact_ibans({
        "account_id": account_id,
        "total_history_count": len(all_rows),
        "returned_count": len(serialized_txs),
        "baseline_avg_amount": round(avg_amount, 2),
        "baseline_median_amount": round(med_amount, 2),
        "max_amount_deviation_ratio": round(max_deviation_ratio, 2),
        "channel_switches": channel_switches,
        "location_changes": location_changes,
        "foreign_tx_count": foreign_tx_count,
        "foreign_countries": sorted(foreign_countries),
        "night_tx_count": night_tx_count,
        "micro_tx_burst_count": micro_burst_count,
        "high_value_tx_count": high_value_count,
        "transactions": serialized_txs,
    })


def get_network_connections(
    account_id: str,
    *,
    engine: Engine | None = None,
    extractor: FeatureExtractor | None = None,
    reference_time: datetime | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Ortak cihaz, `/24` IP alt ağı, fan-in/fan-out yoğunluğu ve döngüsel transfer komşuluklarını döndürür."""
    cfg = settings or get_settings()
    db_engine = engine or get_engine()
    id_cap = cfg.agent.max_evidence_ids

    with session_scope(db_engine) as session:
        acc = session.get(Account, account_id)
        in_extractor = extractor is not None and (
            account_id in extractor.graph.flow or ("A", account_id) in extractor.graph.links
        )
        if acc is None and not in_extractor:
            raise KeyError(f"Hesap bulunamadı: {account_id}")

        ref_ts = _resolve_ref_time(session, account_id, reference_time) or datetime.now(UTC).replace(tzinfo=None)
        lookback_start = ref_ts - timedelta(days=cfg.agent.history_lookback_days)
        window_start = ref_ts - timedelta(seconds=cfg.features.flow_window_seconds)

        # 1) Ortak cihaz paylaşımı (AccountDevice + Device)
        own_devices = [
            str(d)
            for d in session.scalars(
                select(AccountDevice.device_id).where(AccountDevice.account_id == account_id)
            ).all()
        ]
        # İşlemlerde kullanılmış ancak linklenmemiş cihazları da ekle (point-in-time)
        tx_devices = [
            str(d)
            for d in session.scalars(
                select(Transaction.device_id)
                .where(
                    Transaction.sender_account_id == account_id,
                    Transaction.device_id.is_not(None),
                    Transaction.timestamp <= ref_ts,
                )
                .distinct()
            ).all()
        ]
        all_dev_ids = sorted(set(own_devices) | set(tx_devices))

        shared_devices: list[dict[str, Any]] = []
        shared_device_accounts: set[str] = set()
        if all_dev_ids:
            dev_meta = {
                d.device_id: d
                for d in session.scalars(select(Device).where(Device.device_id.in_(all_dev_ids))).all()
            }
            link_rows = session.execute(
                select(AccountDevice.device_id, AccountDevice.account_id).where(
                    AccountDevice.device_id.in_(all_dev_ids)
                )
            ).all()
            tx_link_rows = session.execute(
                select(Transaction.device_id, Transaction.sender_account_id)
                .where(Transaction.device_id.in_(all_dev_ids), Transaction.timestamp <= ref_ts)
                .distinct()
            ).all()

            accounts_by_dev: dict[str, set[str]] = {d: set() for d in all_dev_ids}
            for dev_id, acc_id in (*link_rows, *tx_link_rows):
                if dev_id is not None:
                    accounts_by_dev.setdefault(str(dev_id), set()).add(str(acc_id))

            for dev_id in all_dev_ids:
                others = sorted(a for a in accounts_by_dev.get(dev_id, set()) if a != account_id)
                if others:
                    shared_device_accounts.update(others)
                d_obj = dev_meta.get(dev_id)
                shared_devices.append({
                    "device_id": dev_id,
                    "device_type": d_obj.device_type.value if d_obj else None,
                    "is_emulator": bool(d_obj.is_emulator) if d_obj else False,
                    "shared_accounts": others,
                    "account_count": len(others) + 1,
                })

        # 2) Hesabın tüm geçmişindeki para akışı ve `/24` IP bloğu eşleşmeleri
        acc_txs = session.execute(
            select(
                Transaction.transaction_id,
                Transaction.timestamp,
                Transaction.sender_account_id,
                Transaction.receiver_account_id,
                Transaction.amount_minor,
                Transaction.ip_address,
            )
            .where(
                or_(
                    Transaction.sender_account_id == account_id,
                    Transaction.receiver_account_id == account_id,
                ),
                Transaction.timestamp >= lookback_start,
                Transaction.timestamp <= ref_ts,
            )
            .order_by(Transaction.timestamp.asc())
        ).all()

        own_ip_blocks = {
            blk
            for row in acc_txs
            if row.sender_account_id == account_id and (blk := ip_block(row.ip_address)) is not None
        }

        shared_ip_blocks: list[dict[str, Any]] = []
        shared_ip_accounts: set[str] = set()
        if own_ip_blocks:
            ip_candidates = session.execute(
                select(
                    Transaction.transaction_id,
                    Transaction.sender_account_id,
                    Transaction.ip_address,
                ).where(
                    Transaction.ip_address.is_not(None),
                    Transaction.timestamp >= lookback_start,
                    Transaction.timestamp <= ref_ts,
                )
            ).all()
            accs_by_blk: dict[str, set[str]] = {b: set() for b in own_ip_blocks}
            txs_by_blk: dict[str, list[str]] = {b: [] for b in own_ip_blocks}
            for tx_id, s_acc, ip_val in ip_candidates:
                blk = ip_block(ip_val)
                if blk in own_ip_blocks:
                    accs_by_blk[blk].add(str(s_acc))
                    if len(txs_by_blk[blk]) < id_cap:
                        txs_by_blk[blk].append(str(tx_id))

            for blk in sorted(own_ip_blocks):
                others = sorted(a for a in accs_by_blk[blk] if a != account_id)
                if others:
                    shared_ip_accounts.update(others)
                shared_ip_blocks.append({
                    "ip_block": f"{blk}.0/24",
                    "shared_accounts": others,
                    "account_count": len(others) + 1,
                    "evidence_tx_ids": txs_by_blk[blk],
                })

        # 3) Dereceler ve son 24 saatlik fan-in / fan-out metrikleri
        in_senders = {row.sender_account_id for row in acc_txs if row.receiver_account_id == account_id}
        out_receivers = {row.receiver_account_id for row in acc_txs if row.sender_account_id == account_id}
        reciprocal = sorted(in_senders & out_receivers)

        fan_in_rows = [
            row for row in acc_txs if row.receiver_account_id == account_id and row.timestamp >= window_start
        ]
        fan_out_rows = [
            row for row in acc_txs if row.sender_account_id == account_id and row.timestamp >= window_start
        ]
        fan_in_amt = sum(int(r.amount_minor) for r in fan_in_rows) / 100.0
        fan_out_amt = sum(int(r.amount_minor) for r in fan_out_rows) / 100.0

        # 4) Kısa döngü (cycle) analizi: son 24 saatlik para akışı alt grafında BFS
        recent_edges = session.execute(
            select(
                Transaction.transaction_id,
                Transaction.sender_account_id,
                Transaction.receiver_account_id,
                Transaction.amount_minor,
            )
            .where(
                Transaction.timestamp >= window_start,
                Transaction.timestamp <= ref_ts,
            )
            .order_by(Transaction.timestamp.asc())
        ).all()

    flow_graph = nx.DiGraph()
    for tx_id, s_acc, r_acc, amt_minor in recent_edges:
        if flow_graph.has_edge(s_acc, r_acc):
            flow_graph[s_acc][r_acc]["tx_ids"].append(tx_id)
            flow_graph[s_acc][r_acc]["amount"] += float(amt_minor) / 100.0
        else:
            flow_graph.add_edge(s_acc, r_acc, tx_ids=[tx_id], amount=float(amt_minor) / 100.0)

    cycle_paths: list[list[str]] = []
    cycle_tx_ids: list[str] = []
    max_hops = cfg.features.max_cycle_hops
    if account_id in flow_graph:
        for succ in sorted(flow_graph.successors(account_id)):
            if succ == account_id:
                continue
            try:
                path = nx.shortest_path(flow_graph, source=succ, target=account_id)
            except nx.NetworkXNoPath:
                continue
            full_cycle = [account_id, *path]
            if len(full_cycle) - 1 <= max_hops + 1:
                cycle_paths.append(full_cycle)
                for u, v in itertools.pairwise(full_cycle):
                    edge_txs = flow_graph[u][v]["tx_ids"]
                    if edge_txs:
                        cycle_tx_ids.append(str(edge_txs[-1]))
                if len(cycle_paths) >= cfg.agent.max_cycle_paths:
                    break

    in_deg = len(in_senders)
    out_deg = len(out_receivers)
    return redact_ibans({
        "account_id": account_id,
        "in_degree": in_deg,
        "out_degree": out_deg,
        "in_out_ratio": round((in_deg + 1) / (out_deg + 1), 3),
        "fan_in_24h_unique_senders": len({r.sender_account_id for r in fan_in_rows}),
        "fan_in_24h_tx_count": len(fan_in_rows),
        "fan_in_24h_amount": round(fan_in_amt, 2),
        "fan_in_tx_ids": [str(r.transaction_id) for r in fan_in_rows[:id_cap]],
        "fan_out_24h_unique_receivers": len({r.receiver_account_id for r in fan_out_rows}),
        "fan_out_24h_tx_count": len(fan_out_rows),
        "fan_out_24h_amount": round(fan_out_amt, 2),
        "fan_out_tx_ids": [str(r.transaction_id) for r in fan_out_rows[:id_cap]],
        "fan_out_to_fan_in_ratio": round(fan_out_amt / fan_in_amt, 3) if fan_in_amt > 0 else 0.0,
        "reciprocal_counterparties": reciprocal,
        "short_cycle_detected": bool(cycle_paths),
        "cycle_paths": cycle_paths,
        "cycle_transaction_ids": sorted(set(cycle_tx_ids)),
        "shared_devices": shared_devices,
        "shared_device_accounts": sorted(shared_device_accounts),
        "shared_ip_blocks": shared_ip_blocks,
        "shared_ip_accounts": sorted(shared_ip_accounts),
    })


def get_alert_evidence(
    alert_id: int,
    *,
    engine: Engine | None = None,
) -> dict[str, Any]:
    """Alarmın risk skoru, `graph_score` oranı, SHAP top-3 gerekçeleri ve işlem bağlamını döndürür."""
    db_engine = engine or get_engine()
    with session_scope(db_engine) as session:
        row = session.execute(
            select(Alert, Transaction)
            .join(Transaction, Alert.transaction_id == Transaction.transaction_id)
            .where(Alert.alert_id == alert_id)
        ).first()
        if row is None:
            raise KeyError(f"Alarm bulunamadı: {alert_id}")
        alert, tx = row
        sender_acc = session.get(Account, tx.sender_account_id)
        receiver_acc = session.get(Account, tx.receiver_account_id)
        device_obj = session.get(Device, tx.device_id) if tx.device_id else None

        reasons = list(alert.reasons or [])
        graph_score = float(alert.graph_score) if alert.graph_score is not None else 0.0
        return redact_ibans({
            "alert_id": int(alert.alert_id),
            "transaction_id": str(alert.transaction_id),
            "status": alert.status.value,
            "created_at": alert.created_at.isoformat() if alert.created_at else None,
            "risk_score": round(float(alert.risk_score), 6),
            "ml_score": round(float(alert.ml_score), 6) if alert.ml_score is not None else None,
            "graph_score": round(graph_score, 4),
            "tabular_score_share": round(max(0.0, 1.0 - graph_score), 4),
            "reasons": reasons,
            "top_reasons_tr": [str(r.get("description", "")) for r in reasons if r.get("description")],
            "transaction": {
                "transaction_id": tx.transaction_id,
                "timestamp": tx.timestamp.isoformat(),
                "sender_account_id": tx.sender_account_id,
                "sender_masked_iban": mask_iban(sender_acc.iban if sender_acc else None),
                "receiver_account_id": tx.receiver_account_id,
                "receiver_masked_iban": mask_iban(receiver_acc.iban if receiver_acc else None),
                "amount_minor": int(tx.amount_minor),
                "amount": round(float(tx.amount_minor) / 100.0, 2),
                "currency": tx.currency,
                "channel": tx.channel.value,
                "transaction_type": tx.transaction_type.value,
                "device_id": tx.device_id,
                "device_type": device_obj.device_type.value if device_obj else None,
                "device_is_emulator": bool(device_obj.is_emulator) if device_obj else False,
                "ip_address": tx.ip_address,
                "ip_block": f"{ip_block(tx.ip_address)}.0/24" if ip_block(tx.ip_address) else None,
                "geo_city": tx.geo_city,
                "geo_country": tx.geo_country,
            },
        })


class InvestigationToolkit:
    """Soruşturma ajanı için araç çağrılarını tek veritabanı/graf bağlamında toplar."""

    def __init__(
        self,
        engine: Engine | None = None,
        extractor: FeatureExtractor | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.engine = engine or get_engine()
        self.extractor = extractor
        self.settings = settings or get_settings()

    def get_alert_evidence(self, alert_id: int) -> dict[str, Any]:
        return get_alert_evidence(alert_id, engine=self.engine)

    def get_account_profile(
        self, account_id: str, *, reference_time: datetime | None = None
    ) -> dict[str, Any]:
        return get_account_profile(
            account_id, engine=self.engine, reference_time=reference_time, settings=self.settings
        )

    def get_recent_transactions(
        self,
        account_id: str,
        limit: int | None = None,
        *,
        reference_time: datetime | None = None,
    ) -> dict[str, Any]:
        return get_recent_transactions(
            account_id,
            limit=limit or self.settings.agent.recent_tx_limit,
            engine=self.engine,
            reference_time=reference_time,
            settings=self.settings,
        )

    def get_network_connections(
        self, account_id: str, *, reference_time: datetime | None = None
    ) -> dict[str, Any]:
        return get_network_connections(
            account_id,
            engine=self.engine,
            extractor=self.extractor,
            reference_time=reference_time,
            settings=self.settings,
        )
