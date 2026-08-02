// Production Render Backend URL
const API_URL = "https://nlp-transformer-system-6.onrender.com"; 
let probabilityChart = null;

// Initialize System on DOM Load
document.addEventListener("DOMContentLoaded", () => {
    checkBackendHealth();
    initScrollReveal();
});

// 1. Health Check
async function checkBackendHealth() {
    const dot = document.getElementById("statusDot");
    const text = document.getElementById("statusText");

    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            dot.className = "status-dot online";
            text.innerText = "API Backend Online (Render)";
        } else {
            throw new Error("API Offline");
        }
    } catch (err) {
        dot.className = "status-dot offline";
        text.innerText = "API Backend Disconnected";
    }
}

// 2. Scroll Reveal Animations
function initScrollReveal() {
    const reveals = document.querySelectorAll(".reveal");
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add("visible");
            }
        });
    }, { threshold: 0.1 });

    reveals.forEach(el => observer.observe(el));
}

// 3. Tab Switching
function switchTab(tabId) {
    document.querySelectorAll(".workspace-card").forEach(card => card.classList.remove("active"));
    document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));

    document.getElementById(tabId).classList.add("active");
    event.currentTarget.classList.add("active");
}

// 4. Slider Parameter Value Updates
function updateVal(elementId, value) {
    document.getElementById(elementId).innerText = value;
}

// 5. Typewriter Text Animation Effect
function typeWriterEffect(elementId, text) {
    const target = document.getElementById(elementId);
    target.innerHTML = "";
    let i = 0;
    
    function type() {
        if (i < text.length) {
            target.innerHTML += text.charAt(i);
            i++;
            setTimeout(type, 15);
        }
    }
    type();
}

// -----------------------------------------------------------------------------
// API REQUEST HANDLERS
// -----------------------------------------------------------------------------

// Classification Request
async function runClassification() {
    const input = document.getElementById("classifyInput").value;
    const spinner = document.getElementById("classifySpinner");
    const resultBox = document.getElementById("classifyResult");
    const chartWrapper = document.getElementById("chartWrapper");

    if (!input.trim()) return alert("Please enter text.");

    spinner.classList.remove("hidden");

    try {
        const res = await fetch(`${API_URL}/predict/classify`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: input, task: "topic" })
        });
        const data = await res.json();

        const categories = ["World", "Sports", "Business", "Sci/Tech"];
        const predCategory = categories[data.predicted_class] || "Unknown";

        resultBox.innerHTML = `
            <div><strong>Predicted Category:</strong> <span style="color: #38bdf8; font-weight:700;">${predCategory}</span></div>
            <div><strong>Model Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%</div>
        `;

        chartWrapper.classList.remove("hidden");
        renderChart(categories, data.probabilities);

    } catch (err) {
        resultBox.innerHTML = `<span style="color: #ef4444;">Error: ${err.message}</span>`;
    } finally {
        spinner.classList.add("hidden");
    }
}

// Render Probability Bar Chart using Chart.js
function renderChart(labels, probabilities) {
    const ctx = document.getElementById("probabilityChart").getContext("2d");

    if (probabilityChart) probabilityChart.destroy();

    probabilityChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Probability",
                data: probabilities,
                backgroundColor: "#38bdf8",
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, grid: { color: "#1e293b" }, ticks: { color: "#94a3b8" } },
                x: { grid: { display: false }, ticks: { color: "#94a3b8" } }
            }
        }
    });
}

// Generation Request
async function runGeneration() {
    const prompt = document.getElementById("generatePrompt").value;
    const temp = parseFloat(document.getElementById("paramTemp").value);
    const topK = parseInt(document.getElementById("paramTopK").value);
    const topP = parseFloat(document.getElementById("paramTopP").value);
    const maxTokens = parseInt(document.getElementById("paramTokens").value);
    const spinner = document.getElementById("generateSpinner");

    if (!prompt.trim()) return alert("Please enter prompt.");

    spinner.classList.remove("hidden");

    try {
        const res = await fetch(`${API_URL}/predict/generate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                prompt: prompt,
                max_tokens: maxTokens,
                temperature: temp,
                top_k: topK,
                top_p: topP
            })
        });
        const data = await res.json();
        typeWriterEffect("generateResult", data.generated_text);
    } catch (err) {
        document.getElementById("generateResult").innerText = `Error: ${err.message}`;
    } finally {
        spinner.classList.add("hidden");
    }
}

// Summarization Request
async function runSummarization() {
    const text = document.getElementById("summarizeInput").value;
    const spinner = document.getElementById("summarizeSpinner");

    if (!text.trim()) return alert("Please enter text.");

    spinner.classList.remove("hidden");

    try {
        const res = await fetch(`${API_URL}/predict/summarize`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text, max_length: 60 })
        });
        const data = await res.json();
        document.getElementById("summarizeResult").innerText = data.summary;
    } catch (err) {
        document.getElementById("summarizeResult").innerText = `Error: ${err.message}`;
    } finally {
        spinner.classList.add("hidden");
    }
}

// Translation Request
async function runTranslation() {
    const text = document.getElementById("translateInput").value;
    const spinner = document.getElementById("translateSpinner");

    if (!text.trim()) return alert("Please enter text.");

    spinner.classList.remove("hidden");

    try {
        const res = await fetch(`${API_URL}/predict/translate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text, target_language: "German" })
        });
        const data = await res.json();
        document.getElementById("translateResult").innerHTML = `<strong style="color: #10b981;">${data.translated_text}</strong>`;
    } catch (err) {
        document.getElementById("translateResult").innerText = `Error: ${err.message}`;
    } finally {
        spinner.classList.add("hidden");
    }
}