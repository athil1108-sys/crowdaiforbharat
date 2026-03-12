// ── Configuration ──
let simulationRunning = true;
let currentZone = "Zone_A";
let updateInterval = null;
let currentView = "map";
let stepCount = 0;
let totalSteps = 0;
let lastAIUpdateStep = -1; // Track which simulation step last triggered an AI refresh
const AI_STEP_INTERVAL = 10; // Refresh AI every 10 simulation steps

const ZONE_INFO = {
    "Zone_A": { name: "North Entrance", cap: 200, color: "#22D3EE", id_prefix: "ZA", name_short: "North" },
    "Zone_B": { name: "Main Concourse", cap: 500, color: "#818CF8", id_prefix: "ZB", name_short: "Main" },
    "Zone_C": { name: "South Exit", cap: 180, color: "#34D399", id_prefix: "ZC", name_short: "South" }
};

// Map Blueprint bounds (aligned with backend)
const _ROOMS = [
    [12, 12, 148, 72, "Registration"], [12, 96, 68, 88, "VIP Lounge"],
    [92, 96, 68, 88, "Atrium"], [12, 196, 148, 76, "Workshop A"],
    [12, 284, 148, 80, "Poster Hall"], [198, 12, 134, 108, "Main Stage"],
    [198, 132, 60, 64, "Booth A"], [270, 132, 62, 64, "Booth B"],
    [198, 208, 134, 76, "Activities"], [198, 296, 134, 68, "STEM Lab"],
    [370, 12, 100, 72, "F&B Court"], [370, 96, 100, 72, "Networking"],
    [370, 180, 100, 80, "Conf Rm 1"], [370, 272, 100, 56, "Conf Rm 2"],
    [370, 340, 100, 56, "Breakout"], [488, 12, 92, 104, "SOW Theatre"],
    [488, 128, 92, 68, "Meeting Rms"], [488, 208, 92, 68, "Sponsor Hub"],
    [488, 288, 92, 56, "Staff Only"], [592, 12, 96, 130, "Tech Pavilion"],
    [592, 154, 96, 110, "Demo Zone"], [592, 276, 96, 120, "Startup Alley"]
];
const _CORRIDORS = [
    [160, 0, 28, 420], [332, 0, 28, 420], [480, 0, 28, 420],
    [0, 84, 700, 24], [0, 196, 700, 24], [0, 284, 700, 24]
];
const _ZONE_BOUNDS = {
    "Zone_A": [0, 0, 700, 108, "#22D3EE", "ZONE A · NORTH ENTRANCE"],
    "Zone_B": [0, 108, 700, 112, "#818CF8", "ZONE B · MAIN CONCOURSE"],
    "Zone_C": [0, 220, 700, 200, "#34D399", "ZONE C · SOUTH EXIT"]
};

// Memory for chart histories
let zoneHistories = { "Zone_A": { d: [], v: [], r: [], tc: [], l: [] }, "Zone_B": { d: [], v: [], r: [], tc: [], l: [] }, "Zone_C": { d: [], v: [], r: [], tc: [], l: [] } };

// ── DOM References ──
const dom = {
    btnToggle: document.getElementById("btn-toggle"),
    btnSidebarToggle: document.getElementById("btn-sidebar-toggle"),
    sidebarIcon: document.getElementById("sidebar-icon"),
    mainSidebar: document.getElementById("main-sidebar"),
    liveIndicator: document.getElementById("live-indicator"),
    navLiveText: document.getElementById("nav-live-text"),
    navStep: document.getElementById("nav-step"),
    navSamples: document.getElementById("nav-samples"),
    navCrit: document.getElementById("nav-crit"),
    scenarios: document.querySelectorAll(".nav-scenario"),
    viewBtns: document.querySelectorAll(".view-btn"),
    zoneCardsWrapper: document.getElementById("zone-cards-wrapper"),
    heatmapImg: document.getElementById("heatmap-img"),
    mapPlotly: document.getElementById('map-plotly'),
    mapZ1: document.getElementById("map-z1"),
    mapZ2: document.getElementById("map-z2"),
    mapZ3: document.getElementById("map-z3"),
    gsLow: document.getElementById("gs-low"),
    gsWarn: document.getElementById("gs-warn"),
    gsCrit: document.getElementById("gs-crit"),
    mapBadge: document.getElementById("map-status-badge"),
    sysBanner: document.getElementById("system-banner"),
    bannerTitle: document.getElementById("banner-title"),
    bannerDesc: document.getElementById("banner-desc"),
    zoneTabs: document.querySelectorAll(".zone-tab"),
    chartsZoneTitle: document.getElementById("charts-zone-title"),
    views: { map: document.getElementById("view-map"), charts: document.getElementById("view-charts") }
};

// ── Chart Configurations ──
const baseLayout = {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    margin: { t: 5, b: 25, l: 35, r: 5 },
    xaxis: { type: 'category', showgrid: true, gridcolor: '#0c1a2c', linecolor: '#0d1e30', tickfont: { family: 'JetBrains Mono', color: '#3d5070', size: 9 }, dtick: 3, title: { text: "TIME (HH:MM)", font: { size: 9, color: '#3d5070' } } },
    yaxis: { showgrid: true, gridcolor: '#0c1a2c', linecolor: '#0d1e30', tickfont: { family: 'JetBrains Mono', color: '#3d5070', size: 9 } },
    showlegend: false, hovermode: "x unified",
    dragmode: false, autosize: true
};

// Initialize Map overlay
function initMapOverlay() {
    const VW = 900, VH = 520;
    const shapes = [];

    // Background
    shapes.push({ type: 'rect', x0: 0, y0: 0, x1: VW, y1: VH, fillcolor: 'rgba(7,17,30,0)', line: { width: 0 } });

    // Scale Corridors and Rooms to the new 900x520 bounds (approximate 1.28x width and 1.23x height multiplier of original 700x420)
    // We adjust the bounds of _ZONE_BOUNDS directly for simplicity as the background heatmap will match.
    const _Z_BOUNDS_SCALED = {
        "Zone_A": [0, 0, 900, 140, "#22D3EE", "ZONE A · NORTH ENTRANCE"],
        "Zone_B": [0, 140, 900, 140, "#818CF8", "ZONE B · MAIN CONCOURSE"],
        "Zone_C": [0, 280, 900, 240, "#34D399", "ZONE C · SOUTH EXIT"]
    };

    // Rooms with scaled offsets (multiply coordinates by ~1.28 w / ~1.23 h)
    const _ROOMS_SCALED = [];
    _ROOMS.forEach(r => {
        _ROOMS_SCALED.push([r[0] * 1.28, r[1] * 1.23, r[2] * 1.28, r[3] * 1.23, r[4]]);
    });

    const _CORRIDORS_SCALED = [];
    _CORRIDORS.forEach(c => {
        _CORRIDORS_SCALED.push([c[0] * 1.28, c[1] * 1.23, c[2] * 1.28, c[3] * 1.23]);
    });

    // Corridors
    _CORRIDORS_SCALED.forEach(c => {
        shapes.push({ type: 'rect', x0: c[0], y0: VH - c[1] - c[3], x1: c[0] + c[2], y1: VH - c[1], fillcolor: 'rgba(9,21,37,0.2)', line: { width: 0 } });
    });
    // Rooms
    _ROOMS_SCALED.forEach(r => {
        shapes.push({ type: 'rect', x0: r[0], y0: VH - r[1] - r[3], x1: r[0] + r[2], y1: VH - r[1], fillcolor: 'rgba(12,29,48,0.1)', line: { color: 'rgba(21, 45, 74, 0.4)', width: 1 } });
    });

    // Zone Bounds
    Object.keys(_Z_BOUNDS_SCALED).forEach(zid => {
        const b = _Z_BOUNDS_SCALED[zid];
        const isSelected = (zid === currentZone);
        shapes.push({
            type: 'rect', x0: b[0] + 1, y0: VH - b[1] - b[3] + 1, x1: b[0] + b[2] - 1, y1: VH - b[1] - 1,
            fillcolor: 'rgba(0,0,0,0)',
            line: { color: b[4], width: isSelected ? 2 : 1, dash: isSelected ? 'solid' : 'dot' }
        });
    });

    const annotations = [];
    _ROOMS_SCALED.forEach(r => {
        annotations.push({
            x: r[0] + r[2] / 2, y: VH - r[1] - r[3] / 2, text: r[4], showarrow: false,
            font: { size: 7, color: 'rgba(22, 42, 66, 0.8)', family: 'Inter' }
        });
    });

    Object.keys(_Z_BOUNDS_SCALED).forEach(zid => {
        const b = _Z_BOUNDS_SCALED[zid];
        annotations.push({
            x: b[0] + 6, y: VH - b[1] - 12, text: b[5], showarrow: false,
            font: { size: 8, color: b[4], family: 'JetBrains Mono' }, xanchor: 'left', yanchor: 'top', opacity: 0.9
        });
    });

    const layout = {
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        margin: { l: 0, r: 0, t: 0, b: 0 },
        xaxis: { range: [0, VW], visible: false, fixedrange: true },
        yaxis: { range: [0, VH], visible: false, fixedrange: true },
        shapes: shapes, annotations: annotations, showlegend: false, hovermode: false, dragmode: false
    };

    Plotly.newPlot('map-plotly', [{ x: [], y: [] }], layout, { displayModeBar: false, responsive: true });
}

// ── Application Logic ──
async function fetchTick() {
    if (!simulationRunning) return;
    try {
        await fetch("/api/tick", { method: "POST" });
        await updateData();
    } catch (e) { console.error("Tick failed", e); }
}

async function updateData() {
    try {
        const res = await fetch("/api/data");
        const data = await res.json();

        stepCount = data.step;
        totalSteps = data.total_steps;

        // Update nav stats
        dom.navStep.innerText = `${stepCount}/${totalSteps}`;
        dom.navCrit.innerText = data.global_status.crit;
        dom.navCrit.style.color = data.global_status.crit > 0 ? "#EF4444" : "#e2e8f0";

        // Global Status Panel
        dom.gsLow.innerText = data.global_status.low;
        dom.gsWarn.innerText = data.global_status.warn;
        dom.gsCrit.innerText = data.global_status.crit;

        // Process Histories
        Object.keys(ZONE_INFO).forEach(zKey => {
            const zd = data.zones[zKey] || {};
            const hist = zoneHistories[zKey];
            if (zd.density !== undefined) {
                hist.d.push(zd.density);
                hist.v.push(zd.velocity);
                hist.r.push(zd.risk_probability);
                hist.tc.push(zd.time_to_congestion !== undefined ? zd.time_to_congestion : 0);
                hist.l.push(data.time_label);

                // Keep last 40 samples
                if (hist.l.length > 40) {
                    hist.d.shift(); hist.v.shift(); hist.r.shift(); hist.tc.shift(); hist.l.shift();
                }
            }
        });

        dom.navSamples.innerText = zoneHistories["Zone_A"].l.length;

        // Heatmap Image
        dom.heatmapImg.src = data.heatmap_base64;

        // Map Overlays
        if (data.zones["Zone_A"]) dom.mapZ1.innerText = data.zones["Zone_A"].density.toFixed(1);
        if (data.zones["Zone_B"]) dom.mapZ2.innerText = data.zones["Zone_B"].density.toFixed(1);
        if (data.zones["Zone_C"]) dom.mapZ3.innerText = data.zones["Zone_C"].density.toFixed(1);

        renderZoneCards(data.zones);

        if (currentView === "map") {
            updateMapHighlights(data.zones);
        } else {
            updateChartsView(data.zones);
        }

        autoRefreshAI(data.zones);

    } catch (e) { console.error("Data fetch failed", e); }
}

function renderZoneCards(zonesObj) {
    dom.zoneCardsWrapper.innerHTML = "";
    Object.keys(ZONE_INFO).forEach(zKey => {
        const info = ZONE_INFO[zKey];
        const zData = zonesObj[zKey] || {};

        const density = zData.density ? zData.density.toFixed(1) : "—";
        const velocity = zData.velocity ? zData.velocity.toFixed(2) : "—";

        let badgeClass = "badge";
        let badgeText = "NOMINAL";
        let dColor = info.color;

        if (zData.risk_level === "red") { badgeClass += " red"; badgeText = "CRITICAL"; dColor = "#EF4444"; }
        else if (zData.risk_level === "yellow") { badgeClass += " yellow"; badgeText = "ELEVATED"; dColor = "#F59E0B"; }

        const card = document.createElement("div");
        card.className = `zcard ${currentZone === zKey ? "active" : ""}`;
        card.innerHTML = `
            <div class="zcard-hdr">
                <div class="zcard-title">
                    <div style="background:${info.color};color:black;font-size:8px;font-family:'JetBrains Mono';padding:2px 4px;border-radius:3px;">${info.id_prefix}</div>
                    ${info.name_short}
                </div>
                <div class="${badgeClass}">${badgeText}</div>
            </div>
            <div class="zcard-metrics">
                <div class="zcard-mblock">
                    <span class="zcard-mlbl">DENSITY</span>
                    <span class="zcard-mval" style="color:${dColor}">${density} <span class="zcard-munit">p/m²</span></span>
                </div>
                <div class="zcard-mblock">
                    <span class="zcard-mlbl">VELOCITY</span>
                    <span class="zcard-mval" style="color:var(--indigo)">${velocity} <span class="zcard-munit">m/s</span></span>
                </div>
            </div>
            ${currentZone === zKey ? `<div class="map-pin"><span class="map-pin-icon">📍</span> Select ${info.name}</div>` : ''}
        `;

        card.onclick = () => {
            currentZone = zKey;
            initMapOverlay(); // Redraw map selection
            updateData(); // Re-render everything
        };
        dom.zoneCardsWrapper.appendChild(card);
    });
}

function updateMapHighlights(zonesObj) {
    // Banner logic for map
    const z = zonesObj[currentZone] || {};
    if (z.risk_level === "red") {
        dom.mapBadge.className = "badge red"; dom.mapBadge.innerText = "CRITICAL";
        dom.sysBanner.style.borderLeftColor = "#EF4444";
        dom.bannerTitle.style.color = "#EF4444";
        dom.bannerTitle.innerText = "SYSTEM CRITICAL";
        dom.bannerDesc.innerText = z.message || "Emergency detected.";
    } else if (z.risk_level === "yellow") {
        dom.mapBadge.className = "badge yellow"; dom.mapBadge.innerText = "ELEVATED";
        dom.sysBanner.style.borderLeftColor = "#F59E0B";
        dom.bannerTitle.style.color = "#F59E0B";
        dom.bannerTitle.innerText = "SYSTEM ELEVATED";
        dom.bannerDesc.innerText = z.message || "Elevated crowding.";
    } else {
        dom.mapBadge.className = "badge"; dom.mapBadge.innerText = "NOMINAL";
        dom.sysBanner.style.borderLeftColor = "#22C55E";
        dom.bannerTitle.style.color = "#22C55E";
        dom.bannerTitle.innerText = "SYSTEM NORMAL";
        dom.bannerDesc.innerText = "All zones operating within normal parameters.";
    }
}

function updateChartsView(zonesObj) {
    const info = ZONE_INFO[currentZone];
    const data = zonesObj[currentZone] || {};
    const hist = zoneHistories[currentZone];

    dom.chartsZoneTitle.innerText = info.name;

    // Sync tabs
    dom.zoneTabs.forEach(t => {
        t.className = `zone-tab ${t.dataset.zone === currentZone ? "active" : ""}`;
    });

    // Header values
    let dColor = info.color;
    if (data.risk_level === "red") dColor = "#EF4444";
    else if (data.risk_level === "yellow") dColor = "#F59E0B";

    document.getElementById("cv-density").innerText = data.density ? data.density.toFixed(2) : "0.00";
    document.getElementById("cv-density").style.color = dColor;
    document.getElementById("cv-velocity").innerText = data.velocity ? data.velocity.toFixed(2) : "0.00";
    document.getElementById("cv-velocity").style.color = "#818CF8";

    const hc = Math.round(data.density * 50); // rough estimation for headcount
    document.getElementById("cv-headcount").innerText = hc.toString();

    // AI Predictions Setup
    document.getElementById("cv-time").innerText = data.time_to_congestion ? (data.time_to_congestion > 0 ? data.time_to_congestion.toFixed(1) : "--") : "--";
    document.getElementById("cv-prob").innerText = data.risk_probability ? data.risk_probability.toFixed(1) + "%" : "0.0%";

    // Plotly Updates
    if (hist.l.length > 1) {
        let xArr = [...hist.l];
        let dArr = [...hist.d];
        let vArr = [...hist.v];
        let tcArr = [...hist.tc];

        renderIndividualChart('chart-density', xArr, dArr, dColor, 'p/m²', 2.5);
        renderIndividualChart('chart-velocity', xArr, vArr, '#818CF8', 'm/s', 0.5);

        let hcHist = hist.d.map(d => Math.round(d * 50));
        renderIndividualChart('chart-headcount', xArr, hcHist, '#22D3EE', 'count', info.cap);

        // Advanced AI Plots
        let riskHist = hist.r.map(r => r / 100);
        renderIndividualChart('chart-time', xArr, tcArr, '#c084fc', 'min', null);
        renderIndividualChart('chart-prob', xArr, riskHist, '#F59E0B', '%', null);
        
        // Force relayout on next frame to ensure 100% width stretch
        requestAnimationFrame(() => { window.dispatchEvent(new Event('resize')); });
    }
}

function renderIndividualChart(id, x, y, color, label, refVal) {
    const trace1 = { x: x, y: y, fill: 'tozeroy', fillcolor: color + '15', type: 'scatter', mode: 'none', hoverinfo: 'skip' };
    const trace2 = { x: x, y: y, type: 'scatter', mode: 'lines', line: { color: color, width: 2, shape: 'linear' }, hoverinfo: 'none' };

    const layout = JSON.parse(JSON.stringify(baseLayout));
    if (label) {
        layout.yaxis.title = { text: label, font: { size: 9, color: '#3d5070' } };
        layout.margin.l = 40; // Extra left margin for label
    }
    
    // Explicitly set x-axis range to prevent clipping on categorical axes
    layout.xaxis.range = [0, x.length - 1];
    
    if (refVal) {
        layout.shapes = [{
            type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: refVal, y1: refVal,
            line: { color: 'rgba(239, 68, 68, 0.5)', width: 1, dash: 'dot' }
        }];
    }

    let yMax = Math.max(...y) * 1.5;
    if (yMax < refVal) yMax = refVal * 1.5;
    layout.yaxis.range = [0, yMax];

    Plotly.react(id, [trace1, trace2], layout, { displayModeBar: false, responsive: true });
}



// ── Controls ──
let sidebarOpen = true;

dom.btnSidebarToggle.onclick = () => {
    sidebarOpen = !sidebarOpen;
    if (sidebarOpen) {
        dom.mainSidebar.classList.remove("collapsed");
        dom.sidebarIcon.innerHTML = '<polyline points="15 18 9 12 15 6"></polyline>'; // point left to close
    } else {
        dom.mainSidebar.classList.add("collapsed");
        dom.sidebarIcon.innerHTML = '<polyline points="9 18 15 12 9 6"></polyline>'; // point right to open
    }
    // Force relayout so Plotly adjusts to the new Main area width
    setTimeout(() => { window.dispatchEvent(new Event('resize')); }, 310);
};

dom.btnToggle.onclick = () => {
    simulationRunning = !simulationRunning;
    dom.btnToggle.innerText = simulationRunning ? "⏸" : "▶";
    if (simulationRunning) {
        dom.navLiveText.innerText = "LIVE";
        dom.liveIndicator.className = "dot-live blink";
    } else {
        dom.navLiveText.innerText = "PAUSED";
        dom.liveIndicator.className = "dot-paused";
    }
};

async function resetSimulation() {
    await fetch("/api/reset", { method: "POST" });
    Object.values(zoneHistories).forEach(h => { h.d = []; h.v = []; h.r = []; h.l = []; });
    updateData();
}

dom.scenarios.forEach(btn => {
    btn.onclick = async () => {
        dom.scenarios.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        Object.values(zoneHistories).forEach(h => { h.d = []; h.v = []; h.r = []; h.l = []; });
        await fetch("/api/scenario", {
            method: "POST", headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scenario: btn.dataset.scenario })
        });
        updateData();
    };
});

dom.viewBtns.forEach(btn => {
    btn.onclick = () => {
        dom.viewBtns.forEach(b => { b.classList.remove('active'); b.style.paddingLeft = ''; });
        btn.classList.add('active');

        currentView = btn.dataset.view;
        dom.views.map.style.display = currentView === 'map' ? 'flex' : 'none';
        dom.views.charts.style.display = currentView === 'charts' ? 'flex' : 'none';

        // Force relayout for Plotly specifically required when unhiding divs
        window.dispatchEvent(new Event('resize'));
        updateData();
    };
});

dom.zoneTabs.forEach(btn => {
    btn.onclick = () => {
        currentZone = btn.dataset.zone;
        initMapOverlay();
        updateData();
    };
});

// ── Bedrock AI Integration ──
async function refreshAI() {
    const aiText = document.getElementById('ai-overview-text');
    aiText.innerHTML = '<div style="text-align: center; color: #607898; margin-top: 60px;"><div class="dot-live blink" style="margin-bottom: 15px; width:12px; height:12px;"></div><br>Synthesizing real-time overview<br>with <b>Amazon Bedrock</b>...</div>';
    try {
        const res = await fetch("/api/overview");
        const data = await res.json();
        // Convert markdown/newlines to HTML
        let formatted = data.overview.replace(/\n/g, '<br>');
        // Simple bolding for "Zone_A:", etc.
        formatted = formatted.replace(/Zone_[A-C]/g, match => `<strong style="color:var(--cyan)">${match}</strong>`);
        aiText.innerHTML = `<div style="animation: fadeIn 0.5s;">${formatted}</div>`;
        lastAIUpdateStep = stepCount;
    } catch (e) {
        aiText.innerHTML = '<div style="color:#EF4444; padding: 20px;">⚠️ <b>Connection Error</b><br><br>Failed to reach AI endpoint. Ensure the backend is running and AWS credentials are valid.</div>';
    }
}

function autoRefreshAI(zones) {
    // Trigger AI refresh every 10 steps (e.g., step 10, 20, 30...)
    // Also trigger on first step if it hasn't run yet
    const shouldRefresh = (stepCount > 0 && (stepCount % AI_STEP_INTERVAL === 0)) || (lastAIUpdateStep === -1 && stepCount > 0);
    
    // Ensure we only trigger once per target step
    if (shouldRefresh && stepCount !== lastAIUpdateStep) {
        refreshAI();
    }
}

document.getElementById('btn-refresh-ai').onclick = refreshAI;

// Start loop
initMapOverlay();
updateData();
updateInterval = setInterval(fetchTick, 2000);
