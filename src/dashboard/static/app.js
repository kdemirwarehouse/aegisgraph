/* AegisGraph SOC Dashboard — SSE, Cytoscape.js ve Human-in-the-Loop Kontrolcüsü */

(() => {
  "use strict";

  const state = {
    items: [],
    itemsByTx: new Map(),
    investigationsByAlert: new Map(),
    filter: "all",
    selectedTxId: null,
    selectedAccountId: null,
    graphHops: 2,
    baseTxCount: 0,
    liveTxCount: 0,
    baseOpenOverflow: 0,
    blockedAmountTry: 0,
    latenciesMs: [],
    alertThreshold: 0.7,
    blockThreshold: 0.9,
    reviewThreshold: 0.5,
    cy: null,
    sse: null,
  };

  const els = {
    sseIndicator: document.getElementById("sse-indicator"),
    sseStatusText: document.getElementById("sse-status-text"),
    kpiTotalTx: document.getElementById("kpi-total-tx"),
    kpiOpenAlerts: document.getElementById("kpi-open-alerts"),
    kpiBlockedAmount: document.getElementById("kpi-blocked-amount"),
    kpiAvgLatency: document.getElementById("kpi-avg-latency"),
    kpiThreshold: document.getElementById("kpi-threshold"),
    toastBanner: document.getElementById("toast-banner"),
    toastInner: document.getElementById("toast-inner"),
    toastMessage: document.getElementById("toast-message"),
    toastClose: document.getElementById("toast-close"),
    streamList: document.getElementById("stream-list"),
    graphCenterBadge: document.getElementById("graph-center-badge"),
    graphMetaSummary: document.getElementById("graph-meta-summary"),
    graphCyclePill: document.getElementById("graph-cycle-pill"),
    graphInspector: document.getElementById("graph-node-inspector"),
    btnCenterSender: document.getElementById("btn-center-sender"),
    btnCenterReceiver: document.getElementById("btn-center-receiver"),
    btnHop1: document.getElementById("btn-hop-1"),
    btnHop2: document.getElementById("btn-hop-2"),
    btnGraphFit: document.getElementById("btn-graph-fit"),
    shapSubtitle: document.getElementById("shap-subtitle"),
    selectedScorePill: document.getElementById("selected-score-pill"),
    shapContainer: document.getElementById("shap-container"),
    btnRunAgent: document.getElementById("btn-run-agent"),
    agentReportContainer: document.getElementById("agent-report-container"),
    alertStatusLabel: document.getElementById("alert-status-label"),
    btnDecisionBlock: document.getElementById("btn-decision-block"),
    btnDecisionFreeze: document.getElementById("btn-decision-freeze"),
    btnDecisionFp: document.getElementById("btn-decision-fp"),
  };

  function escapeHtml(raw) {
    if (raw === null || raw === undefined) return "";
    return String(raw)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatTry(amount) {
    const val = Number(amount || 0);
    return new Intl.NumberFormat("tr-TR", {
      style: "currency",
      currency: "TRY",
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(val);
  }

  function formatInt(num) {
    return new Intl.NumberFormat("tr-TR").format(Number(num || 0));
  }

  function formatShortTime(isoStr) {
    if (!isoStr) return "—";
    const d = new Date(isoStr);
    if (Number.isNaN(d.getTime())) return String(isoStr).slice(11, 19) || String(isoStr);
    return d.toLocaleTimeString("tr-TR", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  function showToast(message, tone = "info") {
    const styles = {
      info: "border-cyan-500/40 bg-cyan-950/60 text-cyan-200",
      success: "border-emerald-500/40 bg-emerald-950/60 text-emerald-200",
      warning: "border-amber-500/40 bg-amber-950/60 text-amber-200",
      error: "border-rose-500/40 bg-rose-950/60 text-rose-200",
    };
    els.toastInner.className = `rounded-lg border px-3 py-2 text-xs font-medium flex items-center justify-between ${styles[tone] || styles.info}`;
    els.toastMessage.textContent = message;
    els.toastBanner.classList.remove("hidden");
  }

  function decideFromScore(score) {
    const s = Number(score || 0);
    if (s >= state.blockThreshold) return "BLOCK";
    if (s >= state.alertThreshold) return "ALERT";
    if (s >= state.reviewThreshold) return "REVIEW";
    return "ALLOW";
  }

  function recomputeKpis() {
    els.kpiTotalTx.textContent = formatInt(state.baseTxCount + state.liveTxCount);

    const openStatuses = new Set(["OPEN", "INVESTIGATING", "ESCALATED"]);
    let openCount = 0;
    let blockedTry = 0;
    for (const item of state.items) {
      const isAlertItem = Boolean(item.is_alert || item.alert_id);
      if (isAlertItem) {
        const status = item.status || "OPEN";
        if (openStatuses.has(status)) openCount += 1;
        if (item.decision === "BLOCK" || item.decision === "ALERT" || status === "CONFIRMED_FRAUD") {
          blockedTry += Number(item.amount || 0);
        }
      }
    }
    const displayOpen = state.baseOpenOverflow + openCount;
    els.kpiOpenAlerts.textContent = formatInt(displayOpen);
    els.kpiBlockedAmount.textContent = formatTry(blockedTry);

    if (state.latenciesMs.length > 0) {
      const avg = state.latenciesMs.reduce((a, b) => a + b, 0) / state.latenciesMs.length;
      els.kpiAvgLatency.textContent = `${avg.toFixed(1)} ms`;
    }
  }

  function upsertStreamItem(raw) {
    const txId = raw.transaction_id;
    if (!txId) return null;
    const prev = state.itemsByTx.get(txId);
    const merged = {
      ...(prev || {}),
      ...raw,
      decision: raw.decision || (prev && prev.decision) || decideFromScore(raw.risk_score),
      is_alert: Boolean(raw.is_alert || (prev && prev.is_alert) || raw.alert_id || (prev && prev.alert_id)),
      ml_score: raw.ml_score !== undefined && raw.ml_score !== null ? raw.ml_score : (prev ? prev.ml_score : raw.risk_score),
      graph_score: raw.graph_score !== undefined && raw.graph_score !== null ? raw.graph_score : (prev ? prev.graph_score : 0),
    };
    state.itemsByTx.set(txId, merged);
    if (!prev) {
      state.items.unshift(merged);
      if (state.items.length > 150) {
        const removed = state.items.pop();
        if (removed) state.itemsByTx.delete(removed.transaction_id);
      }
    } else {
      const idx = state.items.findIndex((x) => x.transaction_id === txId);
      if (idx >= 0) state.items[idx] = merged;
    }
    return merged;
  }

  function renderStream() {
    const filtered = state.items.filter((item) => {
      if (state.filter === "alerts") return Boolean(item.is_alert || item.alert_id);
      if (state.filter === "investigations") {
        return Boolean(item.alert_id && state.investigationsByAlert.has(item.alert_id));
      }
      return true;
    });

    if (filtered.length === 0) {
      els.streamList.innerHTML =
        '<div class="text-center text-xs text-slate-500 py-10">Bu filtrede henüz kayıt yok.</div>';
      return;
    }

    els.streamList.innerHTML = filtered
      .slice(0, 80)
      .map((item) => {
        const isSelected = item.transaction_id === state.selectedTxId;
        const graphSharePct = Math.max(0, Math.min(100, Math.round(Number(item.graph_score || 0) * 100)));
        const mlSharePct = 100 - graphSharePct;
        const hasInv = item.alert_id && state.investigationsByAlert.has(item.alert_id);
        const sIban = item.sender_masked_iban || "TR** **** **** **** **** **** **";
        const rIban = item.receiver_masked_iban || "TR** **** **** **** **** **** **";
        const riskVal = Number(item.risk_score || 0);
        const mlVal = Number(item.ml_score !== undefined && item.ml_score !== null ? item.ml_score : riskVal);

        return `
          <article data-tx-id="${escapeHtml(item.transaction_id)}" class="stream-card ${isSelected ? "is-selected" : ""}">
            <div class="flex items-center justify-between gap-2">
              <div class="flex items-center gap-1.5 min-w-0">
                <span class="font-mono text-xs font-semibold text-white truncate">${escapeHtml(item.transaction_id)}</span>
                <span class="text-[10px] text-slate-400">${escapeHtml(formatShortTime(item.timestamp || item.transaction_timestamp))}</span>
                ${hasInv ? '<span class="rounded bg-indigo-500/20 border border-indigo-400/40 px-1.5 py-0.5 text-[10px] text-indigo-300">🤖 SAR</span>' : ""}
              </div>
              <span class="badge-decision decision-${escapeHtml(item.decision)}">${escapeHtml(item.decision)}</span>
            </div>

            <div class="mt-1.5 flex items-baseline justify-between gap-2">
              <div class="text-[11px] text-slate-300 truncate">
                <span class="font-mono text-cyan-300">${escapeHtml(item.sender_account_id)}</span>
                <span class="text-slate-500 mx-1">→</span>
                <span class="font-mono text-indigo-300">${escapeHtml(item.receiver_account_id)}</span>
              </div>
              <div class="text-xs font-bold text-white tabular-nums shrink-0">${escapeHtml(formatTry(item.amount))}</div>
            </div>

            <div class="mt-0.5 flex items-center justify-between text-[10px] font-mono text-slate-400">
              <span class="truncate">${escapeHtml(sIban)}</span>
              <span class="mx-1 text-slate-600">→</span>
              <span class="truncate">${escapeHtml(rIban)}</span>
            </div>

            <div class="mt-2">
              <div class="flex items-center justify-between text-[10px] text-slate-400 mb-1">
                <span>ML Skoru: <strong class="text-cyan-300">${mlVal.toFixed(3)}</strong></span>
                <span>Graf Payı: <strong class="text-purple-300">%${graphSharePct}</strong></span>
              </div>
              <div class="h-1.5 w-full overflow-hidden rounded-full bg-slate-800 flex">
                <div class="bg-cyan-500 transition-all duration-300" style="width: ${mlSharePct}%"></div>
                <div class="bg-purple-500 transition-all duration-300" style="width: ${graphSharePct}%"></div>
              </div>
            </div>
          </article>
        `;
      })
      .join("");
  }

  function renderShapPanel(item) {
    if (!item) {
      els.selectedScorePill.textContent = "—";
      els.shapSubtitle.textContent = "Model kararını yukarı çeken ana faktörler";
      els.shapContainer.innerHTML = '<div class="text-xs text-slate-500 py-4 text-center">Seçili işlem yok.</div>';
      return;
    }

    const score = Number(item.risk_score || 0);
    els.selectedScorePill.textContent = `Risk: ${score.toFixed(3)}`;
    els.shapSubtitle.textContent = `${item.transaction_id} • ${item.channel || "—"}`;

    const reasons = Array.isArray(item.reasons) ? item.reasons : [];
    if (reasons.length === 0) {
      els.shapContainer.innerHTML =
        '<div class="text-xs text-slate-400 py-3">Bu işlem için riski artıran pozitif SHAP faktörü bulunmuyor.</div>';
      return;
    }

    const maxShap = Math.max(...reasons.map((r) => Math.abs(Number(r.shap_value || 0))), 0.1);
    els.shapContainer.innerHTML = reasons
      .slice(0, 3)
      .map((reason) => {
        const val = Number(reason.shap_value || 0);
        const pct = Math.min(100, Math.max(8, Math.round((Math.abs(val) / maxShap) * 100)));
        const isGraph = reason.family === "graph";
        const badgeCls = isGraph
          ? "bg-purple-500/20 text-purple-300 border-purple-500/40"
          : "bg-cyan-500/20 text-cyan-300 border-cyan-500/40";
        const barCls = isGraph ? "bg-purple-500" : "bg-cyan-500";
        const familyLabel = isGraph ? "GRAF" : "DAVRANIŞ";
        return `
          <div class="rounded-lg border border-slate-800 bg-slate-950/60 p-2">
            <div class="flex items-start justify-between gap-2 text-xs">
              <span class="text-slate-200 leading-snug">${escapeHtml(reason.description || reason.feature)}</span>
              <span class="shrink-0 font-mono font-semibold text-rose-300">+${val.toFixed(2)}</span>
            </div>
            <div class="mt-1.5 flex items-center gap-2">
              <span class="rounded border px-1.5 py-0.2 text-[10px] font-semibold ${badgeCls}">${familyLabel}</span>
              <div class="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-800">
                <div class="h-full ${barCls} transition-all duration-300" style="width: ${pct}%"></div>
              </div>
            </div>
          </div>
        `;
      })
      .join("");
  }

  function updateAnalystControls(item) {
    const hasAlert = Boolean(item && item.alert_id);
    els.btnRunAgent.disabled = !hasAlert;

    const status = item && item.status ? item.status : hasAlert ? "OPEN" : "—";
    els.alertStatusLabel.textContent = `Durum: ${status}`;

    const isTerminal = status === "CLOSED";
    const enableDecision = hasAlert && !isTerminal;
    els.btnDecisionBlock.disabled = !enableDecision;
    els.btnDecisionFreeze.disabled = !enableDecision;
    els.btnDecisionFp.disabled = !enableDecision;
  }

  function renderInvestigationReport(report) {
    if (!report) {
      els.agentReportContainer.innerHTML =
        '<div class="text-slate-500 py-8 text-center">Bu alarm için henüz SAR raporu oluşturulmamış. Başlatmak için "🤖 Ajanı Başlat" butonunu kullanın.</div>';
      return;
    }

    const evidenceHtml = (report.evidence_chain || [])
      .map((ev) => `<li class="rounded border border-slate-800/80 bg-slate-950/60 px-2.5 py-1.5 text-[11px] text-slate-300 leading-relaxed">${escapeHtml(ev)}</li>`)
      .join("");

    const relatedHtml = (report.related_accounts || [])
      .slice(0, 12)
      .map(
        (acc) =>
          `<button data-graph-acc="${escapeHtml(acc)}" class="related-acc-chip rounded border border-slate-700 bg-slate-800/80 px-1.5 py-0.5 font-mono text-[10px] text-cyan-300 hover:border-cyan-400">${escapeHtml(acc)}</button>`
      )
      .join("");

    els.agentReportContainer.innerHTML = `
      <div class="space-y-2.5">
        <div class="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-indigo-500/30 bg-indigo-950/30 p-2.5">
          <div>
            <div class="text-[10px] uppercase tracking-wider text-indigo-300">Tespit Edilen Tipoloji</div>
            <div class="font-bold text-sm text-white">${escapeHtml(report.suspected_typology)}</div>
          </div>
          <div class="text-right">
            <div class="text-[10px] text-slate-400">Güven Skoru</div>
            <div class="font-mono text-sm font-bold text-emerald-400">%${Math.round(Number(report.confidence_score || 0) * 100)}</div>
          </div>
        </div>

        <div class="flex flex-wrap items-center justify-between gap-2 text-[11px]">
          <span class="rounded border border-amber-500/40 bg-amber-500/15 px-2 py-0.5 font-semibold text-amber-300">
            Öneri: ${escapeHtml(report.recommended_action)}
          </span>
          <span class="rounded border border-slate-700 bg-slate-800 px-2 py-0.5 text-slate-300">
            ${report.requires_human_approval ? "👤 İnsan Analist Onayı Zorunlu" : "Otomatik"}
          </span>
        </div>

        <div class="rounded-lg border border-slate-800 bg-slate-950/70 p-2.5">
          <div class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-slate-400">MASAK / SAR Yönetici Özeti</div>
          <p class="text-xs leading-relaxed text-slate-200">${escapeHtml(report.executive_summary_tr)}</p>
        </div>

        <div>
          <div class="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-slate-400">Kanıt Zinciri (${(report.evidence_chain || []).length})</div>
          <ul class="space-y-1.5">${evidenceHtml}</ul>
        </div>

        <div>
          <div class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-slate-400">İlişkili Hesaplar (Graf Odakla)</div>
          <div class="flex flex-wrap gap-1">${relatedHtml}</div>
        </div>
      </div>
    `;
  }

  function initCytoscape() {
    if (state.cy || typeof window.cytoscape !== "function") return;
    state.cy = window.cytoscape({
      container: document.getElementById("cy"),
      style: [
        {
          selector: 'node[node_type="account"]',
          style: {
            shape: "ellipse",
            "background-color": "#3b82f6",
            "border-color": "#93c5fd",
            "border-width": 2,
            width: 42,
            height: 42,
            label: "data(displayLabel)",
            color: "#f8fafc",
            "font-size": 10,
            "font-weight": 600,
            "text-valign": "bottom",
            "text-margin-y": 5,
            "text-outline-color": "#020617",
            "text-outline-width": 2,
          },
        },
        {
          selector: "node[?is_center]",
          style: {
            "background-color": "#ef4444",
            "border-color": "#fecaca",
            "border-width": 3,
            width: 54,
            height: 54,
            "font-size": 11,
            "font-weight": 700,
          },
        },
        {
          selector: 'node[node_type="device"]',
          style: {
            shape: "hexagon",
            "background-color": "#a855f7",
            "border-color": "#e9d5ff",
            "border-width": 2,
            width: 38,
            height: 38,
            label: "data(displayLabel)",
            color: "#e9d5ff",
            "font-size": 9,
            "text-valign": "bottom",
            "text-margin-y": 4,
            "text-outline-color": "#020617",
            "text-outline-width": 2,
          },
        },
        {
          selector: 'node[node_type="ip_block"]',
          style: {
            shape: "rectangle",
            "background-color": "#64748b",
            "border-color": "#cbd5e1",
            "border-width": 1.5,
            width: 34,
            height: 28,
            label: "data(displayLabel)",
            color: "#cbd5e1",
            "font-size": 9,
            "text-valign": "bottom",
            "text-margin-y": 4,
            "text-outline-color": "#020617",
            "text-outline-width": 2,
          },
        },
        {
          selector: 'edge[edge_type="money_flow"]',
          style: {
            width: 2.2,
            "line-color": "#38bdf8",
            "target-arrow-color": "#38bdf8",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            label: "data(edgeLabel)",
            "font-size": 9,
            color: "#bae6fd",
            "text-background-color": "#020617",
            "text-background-opacity": 0.85,
            "text-background-padding": 2,
          },
        },
        {
          selector: 'edge[edge_type="uses_device"]',
          style: {
            width: 1.6,
            "line-color": "#a855f7",
            "line-style": "dashed",
            "curve-style": "bezier",
          },
        },
        {
          selector: 'edge[edge_type="uses_ip_block"]',
          style: {
            width: 1.4,
            "line-color": "#64748b",
            "line-style": "dotted",
            "curve-style": "bezier",
          },
        },
      ],
      wheelSensitivity: 0.25,
    });

    state.cy.on("tap", "node", (evt) => {
      const data = evt.target.data();
      showNodeInspector(data);
    });

    state.cy.on("dbltap", 'node[node_type="account"]', (evt) => {
      const accId = evt.target.data("id");
      if (accId) loadGraph(accId, state.graphHops);
    });
  }

  function showNodeInspector(data) {
    if (!data) return;
    let html = "";
    if (data.node_type === "account") {
      html = `
        <div class="flex flex-wrap items-center justify-between gap-2">
          <div>
            <span class="font-mono font-bold text-white">${escapeHtml(data.id)}</span>
            <span class="ml-2 text-slate-400">${escapeHtml(data.masked_iban || "")}</span>
          </div>
          <span class="text-[11px] text-cyan-300">Çift tıkla: Merkez Yap</span>
        </div>
        <div class="mt-1 flex flex-wrap gap-3 text-[11px] text-slate-300">
          <span>Tip: <strong>${escapeHtml(data.account_type || "—")}</strong></span>
          <span>KYC: <strong>${escapeHtml(data.kyc_level ?? "—")}/3</strong></span>
          <span>In/Out: <strong>${escapeHtml(data.in_degree ?? 0)}/${escapeHtml(data.out_degree ?? 0)}</strong></span>
          <span>PageRank: <strong>${escapeHtml(data.pagerank ?? 0)}</strong></span>
          <span>Alarm: <strong>${escapeHtml(data.alert_count ?? 0)}</strong></span>
        </div>
      `;
    } else if (data.node_type === "device") {
      html = `
        <div class="flex items-center justify-between">
          <span class="font-mono font-bold text-purple-300">${escapeHtml(data.id)}</span>
          <span class="text-[11px] ${data.is_emulator ? "text-rose-300 font-semibold" : "text-slate-300"}">
            ${data.is_emulator ? "⚠️ Emülatör Cihaz" : escapeHtml(data.device_type || "Cihaz")}
          </span>
        </div>
        <div class="mt-1 text-[11px] text-slate-300">Ortak Kullanan Hesap Sayısı: <strong>${escapeHtml(data.shared_account_count ?? 1)}</strong></div>
      `;
    } else {
      html = `
        <div class="font-mono font-bold text-slate-200">${escapeHtml(data.label)}</div>
        <div class="mt-1 text-[11px] text-slate-300">Aynı /24 Bloğu Kullanan Hesap Sayısı: <strong>${escapeHtml(data.shared_account_count ?? 1)}</strong></div>
      `;
    }
    els.graphInspector.innerHTML = html;
    els.graphInspector.classList.remove("hidden");
  }

  async function loadGraph(accountId, hops = state.graphHops) {
    if (!accountId) return;
    state.selectedAccountId = accountId;
    state.graphHops = hops;
    els.btnHop1.classList.toggle("is-active", hops === 1);
    els.btnHop2.classList.toggle("is-active", hops === 2);
    els.graphCenterBadge.textContent = accountId;
    els.graphInspector.classList.add("hidden");

    try {
      const resp = await fetch(`/graph/${encodeURIComponent(accountId)}?hops=${hops}&max_nodes=80`);
      if (!resp.ok) throw new Error(`Graf yüklenemedi (${resp.status})`);
      const graph = await resp.json();

      els.graphMetaSummary.textContent =
        `${graph.node_count} düğüm • ${graph.edge_count} kenar • Ortak cihaz hesabı: ${graph.shared_device_accounts.length} • Ortak IP hesabı: ${graph.shared_ip_accounts.length}`;
      els.graphCyclePill.classList.toggle("hidden", !graph.short_cycle_detected);

      initCytoscape();
      if (!state.cy) return;

      const elements = [];
      for (const n of graph.nodes) {
        const shortLabel = n.node_type === "account" ? n.id : n.node_type === "device" ? n.id : n.label;
        elements.push({
          group: "nodes",
          data: {
            ...n,
            displayLabel: shortLabel,
          },
        });
      }
      for (const e of graph.edges) {
        const edgeLabel =
          e.edge_type === "money_flow"
            ? `${e.tx_count || 1}x • ${formatTry(e.total_amount || 0)}`
            : "";
        elements.push({
          group: "edges",
          data: {
            ...e,
            id: `${e.edge_type}:${e.source}->${e.target}`,
            edgeLabel,
          },
        });
      }

      state.cy.elements().remove();
      state.cy.add(elements);
      state.cy
        .layout({
          name: "cose",
          animate: false,
          padding: 28,
          nodeRepulsion: 6500,
          idealEdgeLength: 95,
        })
        .run();
      state.cy.fit(undefined, 28);
    } catch (err) {
      els.graphMetaSummary.textContent = err.message || "Graf yüklenemedi.";
    }
  }

  async function loadInvestigationForAlert(alertId) {
    if (!alertId) {
      renderInvestigationReport(null);
      return;
    }
    const cached = state.investigationsByAlert.get(alertId);
    if (cached) {
      renderInvestigationReport(cached);
      return;
    }
    try {
      const resp = await fetch(`/alerts/${alertId}/investigation`);
      if (resp.ok) {
        const report = await resp.json();
        state.investigationsByAlert.set(alertId, report);
        renderInvestigationReport(report);
        renderStream();
      } else {
        renderInvestigationReport(null);
      }
    } catch {
      renderInvestigationReport(null);
    }
  }

  async function selectTransaction(txId) {
    const item = state.itemsByTx.get(txId);
    if (!item) return;
    state.selectedTxId = txId;
    renderStream();
    renderShapPanel(item);
    updateAnalystControls(item);

    if (item.sender_account_id) {
      els.btnCenterSender.classList.remove("hidden");
      els.btnCenterSender.textContent = `Gönderici (${item.sender_account_id})`;
    }
    if (item.receiver_account_id) {
      els.btnCenterReceiver.classList.remove("hidden");
      els.btnCenterReceiver.textContent = `Alıcı (${item.receiver_account_id})`;
    }

    const targetAccount = item.sender_account_id || item.receiver_account_id;
    await Promise.all([
      targetAccount ? loadGraph(targetAccount, state.graphHops) : Promise.resolve(),
      loadInvestigationForAlert(item.alert_id),
    ]);
  }

  async function triggerAgentInvestigation() {
    const item = state.itemsByTx.get(state.selectedTxId);
    if (!item || !item.alert_id) return;
    els.btnRunAgent.disabled = true;
    const origText = els.btnRunAgent.textContent;
    els.btnRunAgent.textContent = "⏳ Soruşturuluyor...";
    try {
      const resp = await fetch(`/alerts/${item.alert_id}/investigate`, { method: "POST" });
      if (!resp.ok) throw new Error(`Soruşturma başlatılamadı (${resp.status})`);
      const report = await resp.json();
      state.investigationsByAlert.set(item.alert_id, report);
      if (item.status === "OPEN") {
        item.status = "INVESTIGATING";
      }
      renderInvestigationReport(report);
      updateAnalystControls(item);
      renderStream();
      showToast(`Alarm #${item.alert_id} için SAR raporu oluşturuldu (${report.suspected_typology}).`, "success");
    } catch (err) {
      showToast(err.message || "Ajan çalıştırılamadı.", "error");
    } finally {
      els.btnRunAgent.textContent = origText;
      updateAnalystControls(state.itemsByTx.get(state.selectedTxId));
    }
  }

  async function submitAnalystDecision(newStatus) {
    const item = state.itemsByTx.get(state.selectedTxId);
    if (!item || !item.alert_id) return;
    const noteMap = {
      CONFIRMED_FRAUD: "SOC Dashboard üzerinden analist onayı ile hesap bloke edildi.",
      ESCALATED: "SOC Dashboard üzerinden işlem dondurma ve inceleme yükseltmesi onaylandı.",
      FALSE_POSITIVE: "SOC Dashboard üzerinden analist tarafından yanlış alarm olarak işaretlendi.",
    };
    try {
      const resp = await fetch(`/alerts/${item.alert_id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          status: newStatus,
          assigned_to: "soc_analyst",
          resolution_note: noteMap[newStatus] || "Analist kararı kaydedildi.",
        }),
      });
      if (!resp.ok) {
        const errBody = await resp.json().catch(() => ({}));
        throw new Error(errBody.detail || `Güncelleme başarısız (${resp.status})`);
      }
      const updated = await resp.json();
      upsertStreamItem({ ...updated, is_alert: true });
      const refreshed = state.itemsByTx.get(item.transaction_id);
      updateAnalystControls(refreshed);
      renderStream();
      recomputeKpis();
      showToast(`Alarm #${item.alert_id} durumu "${newStatus}" olarak güncellendi.`, "success");
    } catch (err) {
      showToast(err.message || "Karar kaydedilemedi.", "error");
    }
  }

  async function triggerSimulation(button) {
    const scenario = button.getAttribute("data-scenario") || "ALL";
    const count = Number(button.getAttribute("data-count") || 15);
    button.disabled = true;
    try {
      const resp = await fetch("/simulator/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario, count }),
      });
      if (!resp.ok) {
        const errBody = await resp.json().catch(() => ({}));
        throw new Error(errBody.detail || `Simülatör başlatılamadı (${resp.status})`);
      }
      const body = await resp.json();
      showToast(body.message || `${scenario} simülasyonu başlatıldı.`, "info");
    } catch (err) {
      showToast(err.message || "Simülasyon başlatılamadı.", "error");
    } finally {
      button.disabled = false;
    }
  }

  function handleSseEnvelope(rawEvent, eventType) {
    let envelope;
    try {
      envelope = JSON.parse(rawEvent.data);
    } catch {
      return;
    }
    const payload = envelope && envelope.data ? envelope.data : envelope;
    if (!payload) return;

    if (eventType === "transaction") {
      const existed = state.itemsByTx.has(payload.transaction_id);
      if (!existed) {
        state.liveTxCount += 1;
        if (payload.latency_ms !== undefined) {
          state.latenciesMs.push(Number(payload.latency_ms));
          if (state.latenciesMs.length > 200) state.latenciesMs.shift();
        }
      }
      upsertStreamItem(payload);
      renderStream();
      recomputeKpis();
      if (!state.selectedTxId) {
        selectTransaction(payload.transaction_id);
      }
    } else if (eventType === "alert") {
      upsertStreamItem({ ...payload, is_alert: true });
      renderStream();
      recomputeKpis();
      if (state.selectedTxId === payload.transaction_id) {
        updateAnalystControls(state.itemsByTx.get(payload.transaction_id));
      }
    } else if (eventType === "investigation") {
      if (payload.alert_id) {
        state.investigationsByAlert.set(payload.alert_id, payload);
        const linked = state.items.find((x) => x.alert_id === payload.alert_id);
        if (linked && linked.status === "OPEN") {
          linked.status = "INVESTIGATING";
        }
        renderStream();
        if (linked && state.selectedTxId === linked.transaction_id) {
          renderInvestigationReport(payload);
          updateAnalystControls(linked);
        }
      }
    } else if (eventType === "alert_updated") {
      upsertStreamItem({ ...payload, is_alert: true });
      renderStream();
      recomputeKpis();
      if (state.selectedTxId === payload.transaction_id) {
        updateAnalystControls(state.itemsByTx.get(payload.transaction_id));
      }
    }
  }

  function connectSse() {
    if (state.sse) state.sse.close();
    const sse = new EventSource("/stream/events?replay_recent=25");
    state.sse = sse;

    sse.onopen = () => {
      els.sseIndicator.className = "h-2 w-2 rounded-full bg-emerald-400 animate-pulse";
      els.sseStatusText.textContent = "SSE Live";
    };

    sse.onerror = () => {
      els.sseIndicator.className = "h-2 w-2 rounded-full bg-rose-500";
      els.sseStatusText.textContent = "SSE Yeniden Bağlanıyor...";
    };

    for (const evType of ["transaction", "alert", "investigation", "alert_updated"]) {
      sse.addEventListener(evType, (ev) => handleSseEnvelope(ev, evType));
    }
  }

  async function bootstrap() {
    try {
      const [healthResp, alertsResp] = await Promise.all([
        fetch("/health"),
        fetch("/alerts?limit=50"),
      ]);

      if (healthResp.ok) {
        const health = await healthResp.json();
        state.baseTxCount = Number(health.processed_transactions || 0);
        state.alertThreshold = Number(health.alert_threshold || 0.7);
        state.reviewThreshold = Number(health.review_threshold || 0.5);
        state.blockThreshold = Number(health.block_threshold || 0.9);
        els.kpiThreshold.textContent = state.alertThreshold.toFixed(2);
      }

      if (alertsResp.ok) {
        const alertsData = await alertsResp.json();
        const loaded = alertsData.items || [];
        state.baseOpenOverflow = Math.max(0, Number(alertsData.total || 0) - loaded.length);
        for (const alert of loaded.slice().reverse()) {
          upsertStreamItem({
            ...alert,
            is_alert: true,
            timestamp: alert.transaction_timestamp || alert.created_at,
          });
        }
      }

      renderStream();
      recomputeKpis();
      if (state.items.length > 0 && !state.selectedTxId) {
        await selectTransaction(state.items[0].transaction_id);
      }
    } catch {
      showToast("Başlangıç verisi alınamadı; SSE akışı bekleniyor.", "warning");
    }

    connectSse();
  }

  // Olay Dinleyicileri
  document.querySelectorAll(".sim-btn").forEach((btn) => {
    btn.addEventListener("click", () => triggerSimulation(btn));
  });

  document.querySelectorAll(".filter-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      state.filter = tab.getAttribute("data-filter") || "all";
      document.querySelectorAll(".filter-tab").forEach((t) => t.classList.remove("is-active"));
      tab.classList.add("is-active");
      renderStream();
    });
  });

  els.streamList.addEventListener("click", (evt) => {
    const card = evt.target.closest("[data-tx-id]");
    if (card) {
      selectTransaction(card.getAttribute("data-tx-id"));
    }
  });

  els.agentReportContainer.addEventListener("click", (evt) => {
    const chip = evt.target.closest("[data-graph-acc]");
    if (chip) {
      loadGraph(chip.getAttribute("data-graph-acc"), state.graphHops);
    }
  });

  els.btnHop1.addEventListener("click", () => {
    if (state.selectedAccountId) loadGraph(state.selectedAccountId, 1);
    else {
      state.graphHops = 1;
      els.btnHop1.classList.add("is-active");
      els.btnHop2.classList.remove("is-active");
    }
  });

  els.btnHop2.addEventListener("click", () => {
    if (state.selectedAccountId) loadGraph(state.selectedAccountId, 2);
    else {
      state.graphHops = 2;
      els.btnHop2.classList.add("is-active");
      els.btnHop1.classList.remove("is-active");
    }
  });

  els.btnGraphFit.addEventListener("click", () => {
    if (state.cy) state.cy.fit(undefined, 28);
  });

  els.btnCenterSender.addEventListener("click", () => {
    const item = state.itemsByTx.get(state.selectedTxId);
    if (item && item.sender_account_id) loadGraph(item.sender_account_id, state.graphHops);
  });

  els.btnCenterReceiver.addEventListener("click", () => {
    const item = state.itemsByTx.get(state.selectedTxId);
    if (item && item.receiver_account_id) loadGraph(item.receiver_account_id, state.graphHops);
  });

  els.btnRunAgent.addEventListener("click", triggerAgentInvestigation);
  els.btnDecisionBlock.addEventListener("click", () => submitAnalystDecision("CONFIRMED_FRAUD"));
  els.btnDecisionFreeze.addEventListener("click", () => submitAnalystDecision("ESCALATED"));
  els.btnDecisionFp.addEventListener("click", () => submitAnalystDecision("FALSE_POSITIVE"));
  els.toastClose.addEventListener("click", () => els.toastBanner.classList.add("hidden"));

  bootstrap();
})();
