
const API_BASE = "http://127.0.0.1:8000";

const reviewForm = document.getElementById("reviewForm");
const analyzeButton = document.getElementById("analyzeButton");
const formMessage = document.getElementById("formMessage");
const resultCard = document.getElementById("resultCard");

const historyBody = document.getElementById("historyBody");
const historyCount = document.getElementById("historyCount");

let reviewHistory = [];

function showMessage(message, type = "") {
    formMessage.textContent = message;
    formMessage.className = type;
}

function updateDashboard() {
    const total = reviewHistory.length;

    const approved = reviewHistory.filter(
        review => review.status === "approved"
    ).length;

    const quarantined = reviewHistory.filter(
        review => review.status === "quarantined"
    ).length;

    document.getElementById("totalReviews").textContent = total;
    document.getElementById("approvedReviews").textContent = approved;
    document.getElementById("quarantinedReviews").textContent = quarantined;

    document.getElementById("latestScore").textContent =
        total > 0
            ? (reviewHistory[0].score * 100).toFixed(1) + "%"
            : "—";

    historyCount.textContent =
        `${total} record${total === 1 ? "" : "s"}`;

    if (total === 0) {
        historyBody.innerHTML = `
            <tr>
                <td colspan="5" class="empty-state">
                    No reviews analyzed yet.
                </td>
            </tr>`;
        return;
    }

    historyBody.replaceChildren();

    reviewHistory.forEach(review => {
        const row = document.createElement("tr");

        const values = [
            review.id,
            review.userId,
            review.rating,
            (review.score * 100).toFixed(2) + "%"
        ];

        values.forEach(value => {
            const cell = document.createElement("td");
            cell.textContent = value;
            row.appendChild(cell);
        });

        const statusCell = document.createElement("td");
        const badge = document.createElement("span");

        badge.className =
            "status-badge" +
            (review.status === "quarantined" ? " quarantined" : "");

        badge.textContent = review.status;
        statusCell.appendChild(badge);
        row.appendChild(statusCell);

        historyBody.appendChild(row);
    });
}

reviewForm.addEventListener("submit", async event => {
    event.preventDefault();

    const text = document.getElementById("reviewText").value.trim();
    const userId = Number(document.getElementById("userId").value);
    const rating = Number(document.getElementById("rating").value);
    const velocity = Number(document.getElementById("velocity").value);

    if (!text || !Number.isInteger(userId) || userId <= 0 ||
        !Number.isInteger(rating) || rating < 1 || rating > 5 ||
        !Number.isInteger(velocity) || velocity < 0) {
        showMessage("Please enter valid review details.", "error-message");
        return;
    }

    analyzeButton.disabled = true;
    analyzeButton.textContent = "Analyzing...";
    showMessage("Connecting to ReviewGuard API...");

    try {
        const response = await fetch(`${API_BASE}/api/reviews/submit`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            body: JSON.stringify({
                user_id: userId,
                text: text,
                rating: rating,
                velocity: velocity
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail
                    ? JSON.stringify(data.detail)
                    : `API error: ${response.status}`
            );
        }

        document.getElementById("resultScore").textContent =
            (data.score * 100).toFixed(2) + "%";

        const statusElement = document.getElementById("resultStatus");
        statusElement.textContent = data.status;
        statusElement.className =
            "status-badge" +
            (data.status === "quarantined" ? " quarantined" : "");

        document.getElementById("inferenceMode").textContent =
            data.inference_mode || "Unknown";

        const explanationList =
            document.getElementById("explanationList");

        explanationList.replaceChildren();

        (data.explanation || []).forEach(reason => {
            const item = document.createElement("li");
            item.textContent = reason;
            explanationList.appendChild(item);
        });

        resultCard.classList.remove("hidden");

        reviewHistory.unshift({
            id: data.id,
            userId: userId,
            rating: rating,
            score: data.score,
            status: data.status
        });

        updateDashboard();

        showMessage("Review analyzed successfully!", "success-message");

    } catch (error) {
        console.error("ReviewGuard API error:", error);

        showMessage(
            "Could not connect to the API. Check that FastAPI is running. " +
            error.message,
            "error-message"
        );
    } finally {
        analyzeButton.disabled = false;
        analyzeButton.textContent = "Analyze Review →";
    }
});

document.getElementById("driftButton").addEventListener("click", async () => {
    const driftResult = document.getElementById("driftResult");
    const driftButton = document.getElementById("driftButton");

    driftButton.disabled = true;
    driftButton.textContent = "Checking...";

    try {
        const response = await fetch(`${API_BASE}/api/drift/status`, {
            headers: {
                "Accept": "application/json"
            }
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        driftResult.replaceChildren();

        const heading = document.createElement("h3");
        heading.textContent = data.status.replaceAll("_", " ").toUpperCase();

        const details = document.createElement("p");
        details.textContent =
            `KS statistic: ${data.ks_statistic ?? "N/A"} | ` +
            `P-value: ${data.p_value ?? "N/A"} | ` +
            `Threshold: ${data.threshold ?? "N/A"} | ` +
            `Reference count: ${data.reference_count ?? "N/A"} | ` +
            `Live count: ${data.live_count ?? "N/A"}`;

        driftResult.append(heading, details);

    } catch (error) {
        console.error("Drift API error:", error);
        driftResult.textContent =
            "Unable to retrieve drift status. Check that FastAPI is running.";
    } finally {
        driftButton.disabled = false;
        driftButton.textContent = "Check Drift Status";
    }
});

updateDashboard();
