document.getElementById("address_carpark").addEventListener("input", async function () {
    const query_carpark = this.value.trim();
    const suggestionsDiv_carpark = document.getElementById("suggestions_carpark");

    if (query_carpark.length < 2) {
        suggestionsDiv_carpark.innerHTML = '';
        return;
    }

    try {
        const response_carpark = await fetch(`/suggest_carpark?query_carpark=${encodeURIComponent(query_carpark)}`);
        const suggestions_carpark = await response_carpark.json();

        suggestionsDiv_carpark.innerHTML = '';

        suggestions_carpark.forEach(address_carpark => {
            const div_carpark = document.createElement("div");
            div_carpark.textContent = address_carpark;
            div_carpark.classList.add("suggestion-item_carpark");

            div_carpark.addEventListener("click", function () {
                document.getElementById("address_carpark").value = address_carpark;
                suggestionsDiv_carpark.innerHTML = ''; // Clear dropdown after selection
            });

            suggestionsDiv_carpark.appendChild(div_carpark);
        });

        suggestionsDiv_carpark.style.display = "block";

    } catch (error) {
        console.error("Error fetching suggestions:", error);
    }
});

// Hide suggestions if user clicks outside
document.addEventListener("click", function (event) {
    if (!event.target.closest("#address_carpark") && !event.target.closest("#suggestions_carpark")) {
        document.getElementById("suggestions_carpark").style.display = "none";
    }
});

document.getElementById("predictionForm_carpark").addEventListener("submit", async function (e) {
    e.preventDefault();

    const address_carpark = document.getElementById("address_carpark").value.trim();
    const date_carpark = document.getElementById("date_carpark").value;
    const time_carpark = document.getElementById("time_carpark").value;

    if (!address_carpark || !date_carpark || !time_carpark) {
        alert("Please fill in all fields before submitting.");
        return;
    }

    try {
        const response_carpark = await fetch('/predict_carpark', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ address_carpark, date_carpark, time_carpark })
        });

        const result_carpark = await response_carpark.json();
        const resultDiv_carpark = document.getElementById("result_carpark");
        const nearbyDiv_carpark = document.getElementById("nearbyCarparks_carpark");
        const nearbyList_carpark = document.getElementById("nearbyList_carpark");

        nearbyList_carpark.innerHTML = "";
        nearbyDiv_carpark.style.display = "none";

        if (result_carpark.error_carpark) {
            resultDiv_carpark.innerHTML = `<span class="error_carpark">❌ Error: ${result_carpark.error_carpark}</span>`;
        } else {
            resultDiv_carpark.innerHTML = `<span class="success_carpark">Predicted Available Lots: ${result_carpark.prediction_carpark}</span>`;

            if (result_carpark.prediction_carpark < 20 && result_carpark.nearby_carparks_carpark.length > 0) {
                nearbyDiv_carpark.style.display = "block";
                nearbyList_carpark.innerHTML = `<p class="warning_carpark">⚠️ Oops! Not enough parking spaces here. Consider these nearby carparks instead:</p>`;

                result_carpark.nearby_carparks_carpark.forEach(carpark => {
                    const googleSearchUrl_carpark = `https://www.google.com/maps/search/${encodeURIComponent(carpark.address_carpark)}`;
                    const li_carpark = document.createElement("li");
                    li_carpark.innerHTML = `<a href="${googleSearchUrl_carpark}" target="_blank" class="carpark-address_carpark">${carpark.address_carpark}</a> 
                                            <span class="available-lots_carpark">(Available Lots: ${carpark.available_lots_carpark})</span>`;
                    nearbyList_carpark.appendChild(li_carpark);
                });
            }
        }
    } catch (error) {
        console.error("Error fetching prediction:", error);
        alert("An error occurred while fetching predictions.");
    }
});