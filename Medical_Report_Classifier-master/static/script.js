function classifyReport() {

    let text = document.getElementById("reportText").value;

    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("predictionResult").innerText = data.prediction;
    });
}

