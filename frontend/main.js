async function getState() {
    try {
        const res = await fetch("http://localhost:8000/api/state");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        return data;
    } catch (err) {
        console.error("Failed to fetch state:", err);
        return null;
    }
}

function renderEmptyState() {
    const el = document.getElementById("glasselement");

    el.innerHTML = `
        <div class="glass-container empty">
            <div class="empty-message">
                No active face detected.
            </div>
        </div>
    `;
}

function renderFullState(state) {
    const el = document.getElementById("glasselement");

    const g = state.gemini;  // shortcut

    el.innerHTML = `
      <div class="glass-container">

        <div class="identity-section">
          <div class="name">${g.name}</div>
          <div class="subinfo">
            <span class="occupation">${g.occupation}</span>
            <span class="separator">•</span>
            <span class="relationship">${g.relationship}</span>
          </div>
        </div>

        <div class="current-state-section">
          <div class="section-title">Current State</div>
          <div class="current-text">${g.current_state}</div>
        </div>

        <div class="last-points-section">
          <div class="section-title">Last Points</div>
          <ul class="last-points-list">
            ${g.last_points.map(x => `<li>${x}</li>`).join("")}
          </ul>
        </div>

        <div class="convo-points-section">
          <div class="section-title">Suggested Talking Points</div>
          <ul class="convo-points-list">
            ${g.convo_points.map(x => `<li>${x}</li>`).join("")}
          </ul>
        </div>

      </div>
    `;
}


function renderBasedOnState(state) {
    if (!state) return;

    const isFull =
        state.active_face !== null &&
        state.has_gemini === true;

    if (isFull) {
        renderFullState(state);
    } else {
        renderEmptyState();
    }
}

async function main() {
    while (true) {
        const state = await getState();
        console.log("STATE:", state);

        renderBasedOnState(state);
        await new Promise(resolve => setTimeout(resolve, 200));
    }
}

main();
