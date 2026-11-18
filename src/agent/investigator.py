"""Otonom finansal soruşturma ajanı ve MASAK/SAR rapor sentezleyicisi.

Ajan yalnızca salt-okunur kanıt araçlarını çağırarak delil zinciri toplar,
şüpheli tipolojiyi sınıflandırır, Türkçe MASAK/SAR formatında özet üretir ve
**Human-in-the-Loop** ilkesi gereği daima insan analist onayı bekleyen bir
aksiyon önerisi sunar (asla otomatik `CONFIRMED_FRAUD` kararı vermez).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import Engine, select

from config.settings import Settings, get_settings
from src.agent.tools import InvestigationToolkit
from src.database.db import get_engine, session_scope
from src.database.models import (
    Alert,
    AlertStatus,
    Investigation,
    RecommendedAction,
    SuspectedTypology,
    redact_ibans,
)
from src.features.pipeline import FeatureExtractor

logger = logging.getLogger(__name__)

_TYPOLOGY_TR: dict[SuspectedTypology, str] = {
    SuspectedTypology.FRAUD_RING: "Organize Dolandırıcılık Halkası (Fraud Ring / Layering)",
    SuspectedTypology.ACCOUNT_TAKEOVER: "Hesap Ele Geçirme (Account Takeover / ATO)",
    SuspectedTypology.MONEY_MULE: "Katır Hesap Ağı (Money Mule Fan-In / Fan-Out)",
    SuspectedTypology.CARD_TESTING: "Kart Deneme Saldırısı (Card Testing)",
    SuspectedTypology.ANOMALOUS_ACTIVITY: "Şüpheli / Olağandışı İşlem Aktivitesi",
}

_ACTION_TR: dict[RecommendedAction, str] = {
    RecommendedAction.BLOCK_ACCOUNT: "Gönderici/alıcı şüpheli hesapların geçici bloke edilmesi",
    RecommendedAction.FREEZE_TX: "İlgili işlemin beklemeye alınarak (bloke/inceleme) tutarın dondurulması",
    RecommendedAction.ENHANCED_DUE_DILIGENCE: "Müşteriyle teyit ve Artırılmış Müşteri İncelemesi (EDD) uygulanması",
    RecommendedAction.FALSE_POSITIVE_DISMISS: "Yanlış pozitif değerlendirmesi için analist kapanışına sunulması",
}


class InvestigationReport(BaseModel):
    """Otonom soruşturma ajanının yapılandırılmış SAR çıktı şeması."""

    investigation_id: int | None = None
    alert_id: int
    tx_id: str
    created_at: datetime
    suspected_typology: SuspectedTypology
    confidence_score: float = Field(ge=0.0, le=1.0)
    executive_summary_tr: str = Field(min_length=1, max_length=4000)
    evidence_chain: list[str] = Field(min_length=1)
    related_accounts: list[str]
    recommended_action: RecommendedAction
    requires_human_approval: bool = True
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _enforce_human_in_the_loop(self) -> InvestigationReport:
        if not self.requires_human_approval:
            raise ValueError("Human-in-the-Loop kuralı gereği requires_human_approval False olamaz")
        return self


class FraudInvestigatorAgent:
    """Alarmları çok adımlı araç çağrılarıyla soruşturan otonom ajan."""

    def __init__(
        self,
        *,
        engine: Engine | None = None,
        extractor: FeatureExtractor | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.engine = engine or get_engine()
        self.extractor = extractor
        self.toolkit = InvestigationToolkit(
            engine=self.engine,
            extractor=self.extractor,
            settings=self.settings,
        )

    def investigate(self, alert_id: int, *, persist: bool = True) -> InvestigationReport:
        """Verilen `alert_id` için araç zincirini çalıştırır, SAR raporu üretir ve kaydeder."""
        max_steps = self.settings.agent.max_tool_steps
        tool_trace: list[dict[str, Any]] = []

        def _record_step(tool_name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
            tool_trace.append({
                "step": len(tool_trace) + 1,
                "tool": tool_name,
                "arguments": args,
                "status": "ok",
                "result_keys": sorted(result)[: self.settings.agent.max_evidence_ids],
            })

        # Adım 1: Alarm ve tetikleyen işlem kanıtları
        evidence = self.toolkit.get_alert_evidence(alert_id)
        _record_step("get_alert_evidence", {"alert_id": alert_id}, evidence)

        tx_ctx = evidence["transaction"]
        sender = str(tx_ctx["sender_account_id"])
        receiver = str(tx_ctx["receiver_account_id"])
        ref_time = datetime.fromisoformat(str(tx_ctx["timestamp"]))

        sender_profile: dict[str, Any] = {}
        receiver_profile: dict[str, Any] = {}
        sender_recent: dict[str, Any] = {}
        sender_network: dict[str, Any] = {}
        receiver_network: dict[str, Any] = {}

        if len(tool_trace) < max_steps:
            sender_profile = self.toolkit.get_account_profile(sender, reference_time=ref_time)
            _record_step("get_account_profile", {"account_id": sender}, sender_profile)

        if len(tool_trace) < max_steps:
            receiver_profile = self.toolkit.get_account_profile(receiver, reference_time=ref_time)
            _record_step("get_account_profile", {"account_id": receiver}, receiver_profile)

        if len(tool_trace) < max_steps:
            sender_recent = self.toolkit.get_recent_transactions(sender, reference_time=ref_time)
            _record_step("get_recent_transactions", {"account_id": sender}, sender_recent)

        if len(tool_trace) < max_steps:
            sender_network = self.toolkit.get_network_connections(sender, reference_time=ref_time)
            _record_step("get_network_connections", {"account_id": sender}, sender_network)

        if len(tool_trace) < max_steps:
            receiver_network = self.toolkit.get_network_connections(receiver, reference_time=ref_time)
            _record_step("get_network_connections", {"account_id": receiver}, receiver_network)

        typology, confidence, related_accounts, evidence_chain, action = self._synthesize_evidence(
            evidence=evidence,
            sender_profile=sender_profile,
            receiver_profile=receiver_profile,
            sender_recent=sender_recent,
            sender_network=sender_network,
            receiver_network=receiver_network,
        )

        summary_tr = self._build_executive_summary_tr(
            evidence=evidence,
            sender_profile=sender_profile,
            receiver_profile=receiver_profile,
            typology=typology,
            confidence=confidence,
            evidence_chain=evidence_chain,
            related_accounts=related_accounts,
            action=action,
        )

        # Opsiyonel LLM sağlayıcı etkinse yönetici özetini zenginleştir (hata durumunda yerel özet korunur)
        if self.settings.agent.provider != "local" and self.settings.agent.api_key:
            summary_tr = self._maybe_enrich_with_llm(
                base_summary=summary_tr,
                evidence=evidence,
                typology=typology,
                evidence_chain=evidence_chain,
                action=action,
            )

        now = datetime.now(UTC).replace(tzinfo=None)
        report = InvestigationReport(
            alert_id=alert_id,
            tx_id=str(evidence["transaction_id"]),
            created_at=now,
            suspected_typology=typology,
            confidence_score=confidence,
            executive_summary_tr=summary_tr,
            evidence_chain=evidence_chain,
            related_accounts=related_accounts,
            recommended_action=action,
            requires_human_approval=True,
            tool_trace=tool_trace,
        )

        if persist:
            report = self._persist_report(report)

        logger.info(
            "Soruşturma tamamlandı alert_id=%d tx_id=%s tipoloji=%s güven=%.3f öneri=%s",
            report.alert_id,
            report.tx_id,
            report.suspected_typology.value,
            report.confidence_score,
            report.recommended_action.value,
        )
        return report

    def get_report(self, alert_id: int) -> InvestigationReport:
        """Veritabanında kayıtlı soruşturma raporunu döndürür."""
        with session_scope(self.engine) as session:
            row = session.scalars(select(Investigation).where(Investigation.alert_id == alert_id)).first()
            if row is None:
                raise KeyError(f"Alarm için kayıtlı soruşturma raporu bulunamadı: {alert_id}")
            return InvestigationReport(
                investigation_id=int(row.investigation_id),
                alert_id=int(row.alert_id),
                tx_id=str(row.tx_id),
                created_at=row.created_at,
                suspected_typology=row.suspected_typology,
                confidence_score=float(row.confidence_score),
                executive_summary_tr=row.executive_summary_tr,
                evidence_chain=list(row.evidence_chain or []),
                related_accounts=list(row.related_accounts or []),
                recommended_action=row.recommended_action,
                requires_human_approval=bool(row.requires_human_approval),
                tool_trace=list(row.tool_trace or []),
            )

    def _persist_report(self, report: InvestigationReport) -> InvestigationReport:
        with session_scope(self.engine) as session:
            existing = session.scalars(
                select(Investigation).where(Investigation.alert_id == report.alert_id)
            ).first()
            if existing is None:
                inv = Investigation(
                    alert_id=report.alert_id,
                    tx_id=report.tx_id,
                    created_at=report.created_at,
                    suspected_typology=report.suspected_typology,
                    confidence_score=report.confidence_score,
                    executive_summary_tr=report.executive_summary_tr,
                    evidence_chain=report.evidence_chain,
                    related_accounts=report.related_accounts,
                    recommended_action=report.recommended_action,
                    requires_human_approval=True,
                    tool_trace=report.tool_trace,
                )
                session.add(inv)
            else:
                inv = existing
                inv.tx_id = report.tx_id
                inv.suspected_typology = report.suspected_typology
                inv.confidence_score = report.confidence_score
                inv.executive_summary_tr = report.executive_summary_tr
                inv.evidence_chain = report.evidence_chain
                inv.related_accounts = report.related_accounts
                inv.recommended_action = report.recommended_action
                inv.requires_human_approval = True
                inv.tool_trace = report.tool_trace

            # Human-in-the-Loop: asla otomatik CONFIRMED_FRAUD verilmez; yalnızca inceleme notu ve durum güncellenir
            alert = session.get(Alert, report.alert_id)
            if alert is not None:
                if alert.status is AlertStatus.OPEN:
                    alert.status = AlertStatus.INVESTIGATING
                if not alert.assigned_to:
                    alert.assigned_to = "aegis-investigator"
                alert.resolution_note = (
                    f"[Otonom SAR Ön Raporu] Tipoloji: {report.suspected_typology.value} | "
                    f"Öneri: {report.recommended_action.value} (İnsan analist onayı bekleniyor)."
                )

            session.flush()
            return report.model_copy(update={"investigation_id": int(inv.investigation_id)})

    # ------------------------------------------------------------------ #
    # Kanıt sentezleme ve tipoloji skorlaması
    # ------------------------------------------------------------------ #
    def _synthesize_evidence(
        self,
        *,
        evidence: dict[str, Any],
        sender_profile: dict[str, Any],
        receiver_profile: dict[str, Any],
        sender_recent: dict[str, Any],
        sender_network: dict[str, Any],
        receiver_network: dict[str, Any],
    ) -> tuple[SuspectedTypology, float, list[str], list[str], RecommendedAction]:
        tx = evidence["transaction"]
        cfg = self.settings.agent
        min_shared = cfg.ring_min_shared_accounts
        risk_score = float(evidence["risk_score"])
        graph_score = float(evidence["graph_score"])
        shap_features = {str(r.get("feature")) for r in evidence.get("reasons", [])}

        # Tipoloji skorları [0, 1+]
        scores: dict[SuspectedTypology, float] = {
            SuspectedTypology.FRAUD_RING: 0.0,
            SuspectedTypology.MONEY_MULE: 0.0,
            SuspectedTypology.CARD_TESTING: 0.0,
            SuspectedTypology.ACCOUNT_TAKEOVER: 0.0,
        }

        s_shared_dev = sender_network.get("shared_device_accounts", [])
        r_shared_dev = receiver_network.get("shared_device_accounts", [])
        s_shared_ip = sender_network.get("shared_ip_accounts", [])
        r_shared_ip = receiver_network.get("shared_ip_accounts", [])
        has_cycle = bool(sender_network.get("short_cycle_detected") or receiver_network.get("short_cycle_detected"))
        reciprocal = bool(sender_network.get("reciprocal_counterparties"))

        # 1) FRAUD_RING sinyalleri
        if has_cycle:
            scores[SuspectedTypology.FRAUD_RING] += 0.45
        elif reciprocal:
            scores[SuspectedTypology.FRAUD_RING] += 0.20
        if len(s_shared_dev) >= min_shared or len(r_shared_dev) >= min_shared:
            scores[SuspectedTypology.FRAUD_RING] += 0.25
        if len(s_shared_ip) >= min_shared or len(r_shared_ip) >= min_shared:
            scores[SuspectedTypology.FRAUD_RING] += 0.15
        if sender_profile.get("kyc_level", 3) <= 2 and receiver_profile.get("kyc_level", 3) <= 2:
            scores[SuspectedTypology.FRAUD_RING] += 0.10
        ring_shap = {
            "short_cycle_risk",
            "has_reciprocal_flow",
            "shared_device_account_count",
            "shared_ip_account_count",
        }
        if shap_features & ring_shap:
            scores[SuspectedTypology.FRAUD_RING] += 0.15

        # 2) MONEY_MULE sinyalleri
        r_fan_in = int(receiver_network.get("fan_in_24h_unique_senders", 0))
        r_in_deg = int(receiver_network.get("in_degree", 0))
        r_out_ratio = float(receiver_network.get("fan_out_to_fan_in_ratio", 0.0))
        s_fan_in = int(sender_network.get("fan_in_24h_unique_senders", 0))
        s_out_ratio = float(sender_network.get("fan_out_to_fan_in_ratio", 0.0))

        mule_fan_in = cfg.mule_min_fan_in
        if r_fan_in >= 2 * mule_fan_in or r_in_deg >= 2 * mule_fan_in + 2:
            scores[SuspectedTypology.MONEY_MULE] += 0.50
        elif r_fan_in >= mule_fan_in or r_in_deg >= mule_fan_in + 2 or s_fan_in >= mule_fan_in + 1:
            scores[SuspectedTypology.MONEY_MULE] += 0.38
        pt_lo, pt_hi = cfg.mule_passthrough_min_ratio, cfg.mule_passthrough_max_ratio
        hub_min = mule_fan_in - 1
        if (pt_lo <= r_out_ratio <= pt_hi and r_fan_in >= hub_min) or (
            pt_lo <= s_out_ratio <= pt_hi and s_fan_in >= hub_min
        ):
            scores[SuspectedTypology.MONEY_MULE] += 0.28
        if shap_features & {
            "receiver_unique_senders_24h",
            "receiver_in_degree",
            "receiver_out_degree",
            "receiver_in_out_ratio",
            "sender_in_out_ratio",
        }:
            scores[SuspectedTypology.MONEY_MULE] += 0.12
        r_in_out = float(receiver_network.get("in_out_ratio", 1.0))
        if r_in_out >= cfg.mule_hub_in_out_ratio and receiver_profile.get("segment") != "MERCHANT":
            scores[SuspectedTypology.MONEY_MULE] += 0.10

        # 3) CARD_TESTING sinyalleri
        micro_burst = int(sender_recent.get("micro_tx_burst_count", 0))
        is_ecom = tx.get("channel") == "ECOMMERCE" or receiver_profile.get("segment") == "MERCHANT"
        if micro_burst >= cfg.card_testing_strong_burst:
            scores[SuspectedTypology.CARD_TESTING] += 0.55
        elif micro_burst >= cfg.card_testing_min_burst:
            scores[SuspectedTypology.CARD_TESTING] += 0.40
        if is_ecom:
            scores[SuspectedTypology.CARD_TESTING] += 0.20
        if shap_features & {"tx_count_10m", "tx_count_1h", "receiver_merchant_category"}:
            scores[SuspectedTypology.CARD_TESTING] += 0.15

        # 4) ACCOUNT_TAKEOVER sinyalleri
        is_foreign = tx.get("geo_country") != self.settings.features.home_country
        dev_id = tx.get("device_id")
        known_devs = set(sender_profile.get("registered_device_ids", []))
        is_unusual_device = bool(tx.get("device_is_emulator")) or (dev_id is not None and dev_id not in known_devs)
        max_dev_ratio = float(sender_recent.get("max_amount_deviation_ratio", 0.0))
        tx_amount = float(tx.get("amount", 0.0))
        tx_hour = datetime.fromisoformat(str(tx["timestamp"])).hour
        is_night = self.settings.features.night_start_hour <= tx_hour < self.settings.features.night_end_hour

        if is_foreign or int(sender_recent.get("foreign_tx_count", 0)) >= 1:
            scores[SuspectedTypology.ACCOUNT_TAKEOVER] += 0.22
        ato_shap = {"is_new_device_for_account", "device_is_emulator", "is_foreign_or_new_ip"}
        if is_unusual_device or (shap_features & ato_shap):
            scores[SuspectedTypology.ACCOUNT_TAKEOVER] += 0.30
        if max_dev_ratio >= cfg.amount_deviation_ratio or tx_amount >= self.settings.risk.high_value_amount_try:
            scores[SuspectedTypology.ACCOUNT_TAKEOVER] += 0.25
        if is_night:
            scores[SuspectedTypology.ACCOUNT_TAKEOVER] += 0.10
        if receiver_profile.get("segment") == "PERSONAL" and not has_cycle and r_fan_in < 5:
            scores[SuspectedTypology.ACCOUNT_TAKEOVER] += 0.10

        best_typology, best_signal = max(scores.items(), key=lambda kv: kv[1])
        typology = SuspectedTypology.ANOMALOUS_ACTIVITY if best_signal < cfg.min_typology_signal else best_typology

        confidence = round(min(0.99, max(risk_score * 0.65 + min(best_signal, 1.0) * 0.35, risk_score * 0.85)), 4)

        # İlişkili hesaplar listesi
        related: set[str] = {str(tx["sender_account_id"]), str(tx["receiver_account_id"])}
        related.update(s_shared_dev)
        related.update(r_shared_dev)
        related.update(s_shared_ip)
        related.update(r_shared_ip)
        for path in sender_network.get("cycle_paths", []) + receiver_network.get("cycle_paths", []):
            related.update(str(node) for node in path)
        related_accounts = sorted(related)

        # Kaynak ID'li kanıt zinciri
        evidence_chain: list[str] = [
            f"[ALERT-{evidence['alert_id']} / {evidence['transaction_id']}] "
            f"{tx['sender_account_id']} ({tx['sender_masked_iban']}) -> "
            f"{tx['receiver_account_id']} ({tx['receiver_masked_iban']}) | "
            f"Tutar: {tx['amount']:,.2f} {tx['currency']} | Kanal: {tx['channel']} | "
            f"Risk Skoru: {risk_score:.3f} (Graf payı: %{graph_score * 100:.0f})."
        ]
        for reason in evidence.get("reasons", [])[:3]:
            evidence_chain.append(
                f"[SHAP] {reason.get('description', reason.get('feature'))} "
                f"(+{float(reason.get('shap_value', 0.0)):.2f}, aile={reason.get('family', 'tabular')})."
            )

        if sender_profile and receiver_profile:
            evidence_chain.append(
                f"[PROFİL] Gönderici {sender_profile['account_id']} "
                f"({sender_profile['masked_iban']}, KYC {sender_profile['kyc_level']}/3, "
                f"yaş {sender_profile['account_age_days']:.0f} gün, "
                f"geçmiş alarm {sender_profile['past_alert_count']}); "
                f"Alıcı {receiver_profile['account_id']} "
                f"({receiver_profile['masked_iban']}, KYC {receiver_profile['kyc_level']}/3, "
                f"yaş {receiver_profile['account_age_days']:.0f} gün)."
            )

        if max_dev_ratio >= cfg.amount_deviation_ratio or micro_burst >= cfg.card_testing_min_burst or is_foreign:
            recent_ids = [t["transaction_id"] for t in sender_recent.get("transactions", [])[:5]]
            evidence_chain.append(
                f"[DAVRANIŞ] Baz medyana göre maks. tutar sapması {max_dev_ratio:.1f}x, "
                f"mikro-işlem patlaması {micro_burst}, yabancı işlem {sender_recent.get('foreign_tx_count', 0)}, "
                f"kanal değişimi {sender_recent.get('channel_switches', 0)} "
                f"(örnek işlem ID'leri: {', '.join(recent_ids) or evidence['transaction_id']})."
            )

        if has_cycle:
            cycles = sender_network.get("cycle_paths") or receiver_network.get("cycle_paths") or []
            c_txs = sender_network.get("cycle_transaction_ids") or receiver_network.get("cycle_transaction_ids") or []
            fallback_cycle = f"{tx['sender_account_id']} -> {tx['receiver_account_id']}"
            cycle_str = " -> ".join(cycles[0]) if cycles else fallback_cycle
            evidence_chain.append(
                f"[GRAF-DÖNGÜ] Son 24 saatte dairesel para akışı tespit edildi: {cycle_str} "
                f"(kanıt işlem ID'leri: {', '.join(c_txs[:10]) or evidence['transaction_id']})."
            )

        all_shared_devs = sorted(set(s_shared_dev) | set(r_shared_dev))
        if all_shared_devs:
            dev_ids = [
                d["device_id"]
                for d in sender_network.get("shared_devices", []) + receiver_network.get("shared_devices", [])
                if d.get("shared_accounts")
            ]
            evidence_chain.append(
                f"[GRAF-CİHAZ] Ortak cihaz paylaşımı tespit edildi: cihazlar={sorted(set(dev_ids))}, "
                f"bağlı hesaplar={all_shared_devs[:10]}."
            )

        all_shared_ips = sorted(set(s_shared_ip) | set(r_shared_ip))
        if all_shared_ips:
            blk_names = [
                b["ip_block"]
                for b in sender_network.get("shared_ip_blocks", []) + receiver_network.get("shared_ip_blocks", [])
                if b.get("shared_accounts")
            ]
            evidence_chain.append(
                f"[GRAF-IP] Ortak /24 IP alt ağı paylaşımı: bloklar={sorted(set(blk_names))}, "
                f"hesaplar={all_shared_ips[:10]}."
            )

        if r_fan_in >= hub_min or s_fan_in >= hub_min:
            hub_id = receiver_profile.get("account_id") if r_fan_in >= s_fan_in else sender_profile.get("account_id")
            hub_net = receiver_network if r_fan_in >= s_fan_in else sender_network
            evidence_chain.append(
                f"[GRAF-FAN] Hub hesap {hub_id} için son 24 saatte {hub_net.get('fan_in_24h_unique_senders', 0)} "
                f"farklı göndericiden {hub_net.get('fan_in_24h_amount', 0.0):,.2f} TRY giriş, "
                f"{hub_net.get('fan_out_24h_unique_receivers', 0)} alıcıya "
                f"{hub_net.get('fan_out_24h_amount', 0.0):,.2f} TRY çıkış "
                f"(oran={hub_net.get('fan_out_to_fan_in_ratio', 0.0):.2f}, "
                f"örnek TX: {', '.join(hub_net.get('fan_in_tx_ids', [])[:5])})."
            )

        # Önerilen aksiyon (yalnızca öneri — insan onayı zorunlu)
        thresholds = self.settings.risk
        if risk_score >= thresholds.block_score and (
            typology in {SuspectedTypology.FRAUD_RING, SuspectedTypology.ACCOUNT_TAKEOVER, SuspectedTypology.MONEY_MULE}
            or confidence >= cfg.block_min_confidence
        ):
            action = RecommendedAction.BLOCK_ACCOUNT
        elif risk_score >= thresholds.alert_score:
            action = RecommendedAction.FREEZE_TX
        elif risk_score >= thresholds.review_score:
            action = RecommendedAction.ENHANCED_DUE_DILIGENCE
        else:
            action = RecommendedAction.FALSE_POSITIVE_DISMISS

        return typology, confidence, related_accounts, redact_ibans(evidence_chain), action

    def _build_executive_summary_tr(
        self,
        *,
        evidence: dict[str, Any],
        sender_profile: dict[str, Any],
        receiver_profile: dict[str, Any],
        typology: SuspectedTypology,
        confidence: float,
        evidence_chain: list[str],
        related_accounts: list[str],
        action: RecommendedAction,
    ) -> str:
        tx = evidence["transaction"]
        typology_desc = _TYPOLOGY_TR[typology]
        action_desc = _ACTION_TR[action]
        s_iban = sender_profile.get("masked_iban") or tx["sender_masked_iban"]
        r_iban = receiver_profile.get("masked_iban") or tx["receiver_masked_iban"]
        findings = " ".join(evidence_chain[1:4]) if len(evidence_chain) > 1 else evidence_chain[0]

        risk_val = float(evidence["risk_score"])
        graph_pct = float(evidence["graph_score"]) * 100
        summary = (
            f"MASAK Şüpheli İşlem Bildirimi (SAR) Ön Soruşturma Özeti — "
            f"Alarm #{evidence['alert_id']} / İşlem {evidence['transaction_id']}: "
            f"{tx['timestamp']} tarihinde gönderici {tx['sender_account_id']} ({s_iban}) tarafından "
            f"alıcı {tx['receiver_account_id']} ({r_iban}) hesabına {tx['channel']} kanalıyla "
            f"gerçekleştirilen {tx['amount']:,.2f} {tx['currency']} tutarındaki işlem "
            f"{risk_val:.3f} hibrit risk skoru (graf katkısı %{graph_pct:.0f}) "
            f"üretmiştir. Saptanan tipoloji: {typology_desc} (güven skoru: {confidence:.2f}; "
            f"ilişkili {len(related_accounts)} hesap). "
            f"Temel bulgular: {findings} "
            f"Önerilen aksiyon: {action_desc} ({action.value}). "
            f"Not: Bu rapor karar destek önerisidir; "
            f"kesin bloke / CONFIRMED_FRAUD kararı insan analist onayı gerektirir."
        )
        return redact_ibans(summary)[:4000]

    def _maybe_enrich_with_llm(
        self,
        *,
        base_summary: str,
        evidence: dict[str, Any],
        typology: SuspectedTypology,
        evidence_chain: list[str],
        action: RecommendedAction,
    ) -> str:
        """Harici LLM sağlayıcısı açıksa SAR özetini zenginleştirir (hata durumunda yerel özeti korur)."""
        if self.settings.agent.provider != "openai":
            return base_summary
        prompt = (
            "Sen bir banka mali suçlar (MASAK/SAR) kıdemli analistisin. Yalnızca aşağıdaki doğrulanmış "
            "kanıtları kullanarak kısa, resmi Türkçe SAR yönetici özeti yaz (IBAN maskeli kalmalı, karar "
            "insan analist onayına tabidir):\n"
            f"Tipoloji: {typology.value}\nÖneri: {action.value}\n"
            f"Alarm: {evidence['alert_id']} / TX: {evidence['transaction_id']}\n"
            + "\n".join(f"- {item}" for item in evidence_chain)
        )
        try:
            resp = httpx.post(
                f"{self.settings.agent.llm_base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {self.settings.agent.api_key.get_secret_value()}"},
                json={
                    "model": self.settings.agent.model_name,
                    "temperature": self.settings.agent.temperature,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=self.settings.agent.llm_timeout_seconds,
            )
            resp.raise_for_status()
            text = str(resp.json()["choices"][0]["message"]["content"]).strip()
            if not text:
                return base_summary
            # Derinlemesine savunma: LLM ham IBAN üretse bile maskelenir, HITL notu korunur
            text = redact_ibans(text)
            if "insan analist onayı" not in text:
                text = f"{text} Not: Kesin karar insan analist onayı gerektirir."
            return text[:4000]
        except Exception as exc:
            logger.warning("Harici LLM çağrısı başarısız; yerel SAR özetine dönüldü: %s", exc)
            return base_summary
