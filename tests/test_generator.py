"""Sentetik veri motoru testleri: şema, bütünlük, fraud senaryo topolojisi, determinizm, DB."""

from __future__ import annotations

from datetime import timedelta

import networkx as nx
import pandas as pd
import pytest
from pydantic import ValidationError
from sqlalchemy import func, select, text

from config.settings import GeneratorSettings, RiskThresholds
from src.database.db import create_db_engine, dataframe_to_records, init_db, load_table, persist_dataset
from src.database.models import (
    GROUND_TRUTH_COLUMNS,
    RECORD_SCHEMAS,
    Account,
    FraudScenario,
    Transaction,
    record_fields,
)
from src.generator.synthetic_data import (
    SyntheticDataGenerator,
    SyntheticDataset,
    is_valid_tr_iban,
    make_tr_iban,
)

TEST_CONFIG = GeneratorSettings(
    seed=7,
    n_customers=400,
    n_merchants=40,
    simulation_days=14,
    n_fraud_rings=3,
    n_account_takeovers=6,
    n_mule_networks=2,
    n_card_testing=3,
)


@pytest.fixture(scope="module")
def ds() -> SyntheticDataset:
    return SyntheticDataGenerator(TEST_CONFIG).generate()


def _cases(tx: pd.DataFrame, scenario: FraudScenario) -> dict[str, pd.DataFrame]:
    sub = tx[tx["fraud_scenario"] == scenario.value]
    return {case: grp for case, grp in sub.groupby("fraud_case_id")}


def _ring_members(ds: SyntheticDataset, case: str) -> set[str]:
    acc = ds.accounts
    return set(acc.loc[acc["fraud_case_id"] == case, "account_id"])


def _devices_of(ds: SyntheticDataset, account_id: str) -> set[str]:
    ad = ds.account_devices
    return set(ad.loc[ad["account_id"] == account_id, "device_id"])


# --------------------------------------------------------------------------- #
# Şema
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("table", SyntheticDataset.TABLES)
def test_columns_match_record_contract(ds: SyntheticDataset, table: str) -> None:
    assert list(getattr(ds, table).columns) == record_fields(table)


@pytest.mark.parametrize("table", SyntheticDataset.TABLES)
def test_every_row_passes_pydantic_schema(ds: SyntheticDataset, table: str) -> None:
    schema = RECORD_SCHEMAS[table]
    for row in dataframe_to_records(getattr(ds, table)):
        schema.model_validate(row)


def test_schema_rejects_inconsistent_labels(ds: SyntheticDataset) -> None:
    row = dataframe_to_records(ds.transactions.head(1))[0]
    row.update(is_fraud=True, fraud_scenario="NONE", fraud_case_id=None)
    with pytest.raises(ValidationError):
        RECORD_SCHEMAS["transactions"].model_validate(row)


def test_dtypes(ds: SyntheticDataset) -> None:
    tx = ds.transactions
    assert pd.api.types.is_datetime64_any_dtype(tx["timestamp"])
    assert tx["amount_minor"].dtype == "int64"
    assert tx["is_fraud"].dtype == bool
    assert pd.api.types.is_datetime64_any_dtype(ds.accounts["opened_at"])


def test_ground_truth_columns_exist(ds: SyntheticDataset) -> None:
    for table, cols in GROUND_TRUTH_COLUMNS.items():
        assert set(cols) <= set(getattr(ds, table).columns)


# --------------------------------------------------------------------------- #
# Bütünlük
# --------------------------------------------------------------------------- #
def test_primary_keys_unique(ds: SyntheticDataset) -> None:
    assert ds.accounts["account_id"].is_unique
    assert ds.accounts["iban"].is_unique
    assert ds.devices["device_id"].is_unique
    assert ds.devices["fingerprint"].is_unique
    assert ds.transactions["transaction_id"].is_unique
    assert not ds.account_devices.duplicated(["account_id", "device_id"]).any()


def test_referential_integrity(ds: SyntheticDataset) -> None:
    accounts, devices = set(ds.accounts["account_id"]), set(ds.devices["device_id"])
    tx = ds.transactions
    assert set(tx["sender_account_id"]) <= accounts
    assert set(tx["receiver_account_id"]) <= accounts
    assert set(tx["device_id"].dropna()) <= devices
    assert set(ds.account_devices["account_id"]) <= accounts
    assert set(ds.account_devices["device_id"]) <= devices


def test_every_used_device_link_is_recorded(ds: SyntheticDataset) -> None:
    used = ds.transactions.dropna(subset=["device_id"])[["sender_account_id", "device_id"]]
    used = set(map(tuple, used.drop_duplicates().to_numpy()))
    links = set(map(tuple, ds.account_devices[["account_id", "device_id"]].to_numpy()))
    assert used <= links


def test_transaction_invariants(ds: SyntheticDataset) -> None:
    tx = ds.transactions
    start = pd.Timestamp(TEST_CONFIG.start_date)
    end = start + timedelta(days=TEST_CONFIG.simulation_days)
    assert (tx["amount_minor"] > 0).all()
    assert (tx["sender_account_id"] != tx["receiver_account_id"]).all()
    assert tx["timestamp"].between(start, end, inclusive="left").all()
    assert tx["timestamp"].is_monotonic_increasing  # ID'ler kronolojik
    assert (tx["currency"] == "TRY").all()


def test_no_activity_before_account_opened(ds: SyntheticDataset) -> None:
    tx = ds.transactions
    opened = ds.accounts.set_index("account_id")["opened_at"]
    for col in ("sender_account_id", "receiver_account_id"):
        assert (tx["timestamp"] >= tx[col].map(opened)).all()


def test_ibans_are_valid() -> None:
    iban = make_tr_iban("00062", "1234567890123456")
    assert is_valid_tr_iban(iban)
    assert not is_valid_tr_iban(iban[:-1] + str((int(iban[-1]) + 1) % 10))


def test_generated_ibans_pass_checksum(ds: SyntheticDataset) -> None:
    assert ds.accounts["iban"].map(is_valid_tr_iban).all()


def test_fraud_accounts_not_identifiable_by_id_order(ds: SyntheticDataset) -> None:
    acc = ds.accounts.reset_index(drop=True)
    positions = acc.index[acc["is_fraudster"]]
    # Fraud hesapları ID uzayının sonuna yığılmamalı
    assert positions.min() < len(acc) * 0.5


# --------------------------------------------------------------------------- #
# Etiketler ve oranlar
# --------------------------------------------------------------------------- #
def test_label_consistency(ds: SyntheticDataset) -> None:
    tx = ds.transactions
    is_labelled = tx["fraud_scenario"] != FraudScenario.NONE.value
    assert (tx["is_fraud"] == is_labelled).all()
    assert (tx["fraud_case_id"].notna() == is_labelled).all()


def test_all_scenarios_present(ds: SyntheticDataset) -> None:
    present = set(ds.transactions["fraud_scenario"])
    assert {s.value for s in FraudScenario} <= present
    assert len(_cases(ds.transactions, FraudScenario.FRAUD_RING)) == TEST_CONFIG.n_fraud_rings
    assert len(_cases(ds.transactions, FraudScenario.ACCOUNT_TAKEOVER)) == TEST_CONFIG.n_account_takeovers
    assert len(_cases(ds.transactions, FraudScenario.MONEY_MULE)) == TEST_CONFIG.n_mule_networks
    assert len(_cases(ds.transactions, FraudScenario.CARD_TESTING)) == TEST_CONFIG.n_card_testing


def test_fraud_rate_is_imbalanced(ds: SyntheticDataset) -> None:
    rate = ds.transactions["is_fraud"].mean()
    assert 0.001 < rate < 0.08


def test_default_config_fraud_rate_is_bank_realistic() -> None:
    full = SyntheticDataGenerator(GeneratorSettings()).generate()
    rate = full.transactions["is_fraud"].mean()
    assert len(full.transactions) > 50_000
    assert 0.002 < rate < 0.02


def test_hard_negatives_present(ds: SyntheticDataset) -> None:
    """Meşru trafikte de fraud'a benzeyen sinyaller olmalı; aksi halde model kısayol öğrenir."""
    tx, acc = ds.transactions, ds.accounts
    legit = tx[~tx["is_fraud"]]
    assert (legit["geo_country"] != "TR").sum() > 0, "meşru yurt dışı işlem yok"
    honest = set(acc.loc[~acc["is_fraudster"], "account_id"])
    ad = ds.account_devices[ds.account_devices["account_id"].isin(honest)]
    assert (ad.groupby("device_id")["account_id"].nunique() >= 2).any(), "meşru ortak cihaz yok"
    small_bursts = legit[(legit["amount_minor"] <= 6_000) & (legit["channel"] == "ECOMMERCE")]
    assert small_bursts.groupby("sender_account_id").size().max() >= 4
    fraud = tx[tx["is_fraud"]]
    takeover = fraud[fraud["fraud_scenario"].isin(["ACCOUNT_TAKEOVER", "CARD_TESTING"])]
    assert (takeover["geo_country"] == "TR").any(), "yurt içi saldırgan yok"


# --------------------------------------------------------------------------- #
# Senaryo topolojileri
# --------------------------------------------------------------------------- #
def test_fraud_rings_form_a_single_cycle(ds: SyntheticDataset) -> None:
    for case, grp in _cases(ds.transactions, FraudScenario.FRAUD_RING).items():
        members = _ring_members(ds, case)
        assert TEST_CONFIG.ring_size_min <= len(members) <= TEST_CONFIG.ring_size_max
        internal = grp[grp["sender_account_id"].isin(members) & grp["receiver_account_id"].isin(members)]
        g = nx.DiGraph(list(zip(internal["sender_account_id"], internal["receiver_account_id"], strict=True)))
        assert set(g.nodes) == members, case
        assert nx.is_strongly_connected(g), case
        assert any(len(c) == len(members) for c in nx.simple_cycles(g)), case
        # Her üye her turda tam bir kez gönderir → eşit sayıda giden kenar
        assert internal["sender_account_id"].value_counts().nunique() == 1


def test_fraud_rings_share_devices_and_subnet(ds: SyntheticDataset) -> None:
    for case, grp in _cases(ds.transactions, FraudScenario.FRAUD_RING).items():
        members = _ring_members(ds, case)
        shared = set.intersection(*(_devices_of(ds, m) for m in members))
        assert shared, f"{case}: ortak cihaz yok"
        users = set(ds.account_devices.loc[ds.account_devices["device_id"].isin(shared), "account_id"])
        assert users == members, f"{case}: halka cihazı halka dışında kullanılmış"
        internal = grp[grp["sender_account_id"].isin(members)]
        subnets = internal["ip_address"].str.rsplit(".", n=1).str[0]
        assert subnets.nunique() == 1


def test_fraud_rings_move_fast_and_lose_value(ds: SyntheticDataset) -> None:
    for case, grp in _cases(ds.transactions, FraudScenario.FRAUD_RING).items():
        members = _ring_members(ds, case)
        internal = grp[grp["sender_account_id"].isin(members)].sort_values("timestamp")
        gaps = internal["timestamp"].diff().dropna()
        assert gaps.max().total_seconds() <= 45 * 60, case
        assert internal["amount_minor"].is_monotonic_decreasing, case


def test_ring_members_are_new_low_kyc_accounts(ds: SyntheticDataset) -> None:
    rings = ds.accounts[ds.accounts["fraud_case_id"].fillna("").str.startswith("RING-")]
    start = pd.Timestamp(TEST_CONFIG.start_date)
    assert (rings["opened_at"] > start - timedelta(days=60)).all()
    assert (rings["kyc_level"] <= 2).all()


def test_account_takeover_uses_new_device_and_ip(ds: SyntheticDataset) -> None:
    tx = ds.transactions
    legit = tx[~tx["is_fraud"]]
    medians = legit.groupby("sender_account_id")["amount_minor"].median()
    attacker_devices: set[str] = set()
    for case, grp in _cases(tx, FraudScenario.ACCOUNT_TAKEOVER).items():
        victim = grp["sender_account_id"].unique()
        assert len(victim) == 1
        victim = victim[0]
        assert not ds.accounts.set_index("account_id").at[victim, "is_fraudster"]
        device = grp["device_id"].unique()
        assert len(device) == 1
        attacker_devices.add(device[0])
        before = legit[(legit["sender_account_id"] == victim) & (legit["timestamp"] < grp["timestamp"].min())]
        assert device[0] not in set(before["device_id"].dropna()), case
        # Saldırgan IP'si kurbanın geçmişinde hiç görülmemiş (yurt içi proxy de olabilir)
        assert grp["ip_address"].iloc[0] not in set(before["ip_address"].dropna()), case
        assert grp["ip_address"].nunique() == 1
        if victim in medians.index:
            assert grp["amount_minor"].min() > 5 * medians[victim], case
    # Saldırgan cihaz havuzu birden fazla kurbanda tekrar kullanılır (graf sinyali)
    attacker_tx = tx[tx["device_id"].isin(attacker_devices)]
    victims_per_device = attacker_tx.groupby("device_id")["sender_account_id"].nunique()
    assert victims_per_device.max() >= 2


def test_money_mule_fan_in_fan_out(ds: SyntheticDataset) -> None:
    for case, grp in _cases(ds.transactions, FraudScenario.MONEY_MULE).items():
        mules = _ring_members(ds, case)
        fan_in = grp[~grp["sender_account_id"].isin(mules)]
        fan_out = grp[grp["sender_account_id"].isin(mules)]
        hub = fan_in["receiver_account_id"].unique()
        assert len(hub) == 1, case
        assert (fan_out["sender_account_id"] == hub[0]).all()
        assert fan_in["sender_account_id"].nunique() >= TEST_CONFIG.mule_victims_min
        assert fan_out["receiver_account_id"].nunique() >= 3
        assert fan_out["timestamp"].min() > fan_in["timestamp"].max()
        ratio = fan_out["amount_minor"].sum() / fan_in["amount_minor"].sum()
        assert 0.88 <= ratio <= 0.98, case


def test_card_testing_velocity_burst(ds: SyntheticDataset) -> None:
    for case, grp in _cases(ds.transactions, FraudScenario.CARD_TESTING).items():
        grp = grp.sort_values("timestamp")
        assert grp["sender_account_id"].nunique() == 1
        assert (grp["channel"] == "ECOMMERCE").all()
        probes = grp[grp["amount_minor"] <= 2_000]
        big = grp[grp["amount_minor"] >= 500_000]
        assert len(probes) >= 10, case
        assert len(big) >= 1, case
        assert (probes["timestamp"].max() - probes["timestamp"].min()).total_seconds() <= 3600
        assert big["timestamp"].min() > probes["timestamp"].max()


# --------------------------------------------------------------------------- #
# Determinizm
# --------------------------------------------------------------------------- #
def test_same_seed_is_deterministic(ds: SyntheticDataset) -> None:
    again = SyntheticDataGenerator(TEST_CONFIG).generate()
    for table in SyntheticDataset.TABLES:
        pd.testing.assert_frame_equal(getattr(ds, table), getattr(again, table))


def test_different_seed_differs(ds: SyntheticDataset) -> None:
    other = SyntheticDataGenerator(TEST_CONFIG.model_copy(update={"seed": 8})).generate()
    assert not ds.transactions["amount_minor"].equals(other.transactions["amount_minor"])


def test_generator_is_single_use() -> None:
    gen = SyntheticDataGenerator(TEST_CONFIG)
    gen.generate()
    with pytest.raises(RuntimeError):
        gen.generate()


def test_too_few_customers_for_scenarios_raises() -> None:
    cfg = TEST_CONFIG.model_copy(update={"n_customers": 50, "n_mule_networks": 3, "mule_victims_min": 30,
                                          "mule_victims_max": 40})
    with pytest.raises(ValueError, match="yeterli müşteri"):
        SyntheticDataGenerator(cfg).generate()


# --------------------------------------------------------------------------- #
# Konfigürasyon
# --------------------------------------------------------------------------- #
def test_settings_validation() -> None:
    with pytest.raises(ValidationError):
        GeneratorSettings(ring_size_min=9, ring_size_max=4)
    with pytest.raises(ValidationError):
        RiskThresholds(alert_score=0.95, block_score=0.9)


# --------------------------------------------------------------------------- #
# Veritabanı
# --------------------------------------------------------------------------- #
def test_database_roundtrip(ds: SyntheticDataset, tmp_path) -> None:
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'aegis_test.db').as_posix()}")
    try:
        init_db(engine)
        counts = persist_dataset(ds, engine)
        assert counts["transactions"] == len(ds.transactions)
        with engine.connect() as conn:
            n_tx = conn.execute(select(func.count()).select_from(Transaction)).scalar_one()
            n_acc = conn.execute(select(func.count()).select_from(Account)).scalar_one()
            assert (n_tx, n_acc) == (len(ds.transactions), len(ds.accounts))
            assert conn.execute(text("PRAGMA foreign_key_check")).fetchall() == []
            n_fraud = conn.execute(select(func.count()).where(Transaction.is_fraud)).scalar_one()
            assert n_fraud == int(ds.transactions["is_fraud"].sum())
        back = load_table("transactions", engine)
        assert back["amount_minor"].sum() == ds.transactions["amount_minor"].sum()
        assert set(back["fraud_scenario"]) == set(ds.transactions["fraud_scenario"])
        init_db(engine)  # idempotent
    finally:
        engine.dispose()
