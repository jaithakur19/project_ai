const BACKEND_URL = "http://127.0.0.1:5000/predict";

// Global Variables
const now = new Date();
const currentHour = now.getHours();
const currentMinute = now.getMinutes();
const day = new Date().toLocaleString("en-US", { weekday: "long" }).toLowerCase();

// Function to get the current hour in GMT+8
function getCurrentHourGMT8() {
    const gmt8Time = new Date(now.getTime() + 8 * 3600000); // GMT+8 conversion
    return gmt8Time.getUTCHours();
}

// Function to format the hour to 12-hour format with AM/PM
function formatHour(hour) {
    const period = hour >= 12 ? "PM" : "AM";
    const formattedHour = hour % 12 || 12;
    return `${formattedHour}:00 ${period}`;
}

// Check if the current time is within operating hours (5:30 AM to midnight)
function checkOperatingHours() {
    if (currentHour < 5 || (currentHour === 5 && currentMinute < 30) || currentHour === 0) {
        // Clear all line-group elements
        document.getElementById("operating-status").innerHTML = "";
        const lineGroups = document.querySelectorAll(".line-group");
        lineGroups.forEach(group => group.remove());

        // Display "Service Unavailable" card
        displayOperatingHoursCard();
    } else {
        // Start the auto-ping to fetch station statuses
        startAutoPing();
    }
}

// Display the "Service Unavailable" card
function displayOperatingHoursCard() {
    const statusContainer = document.getElementById("status-container");
    statusContainer.innerHTML = "";

    const card = document.createElement("div");
    card.className = "operating-hours-card";
    card.innerHTML = `
        <h2>Service Unavailable</h2>
        <p>It is currently not operating hours for the MRT system (Operating hours are 5:30 AM to midnight).</p>
    `;

    statusContainer.appendChild(card);

    // Center the card in the container
    statusContainer.style.display = "flex";
    statusContainer.style.justifyContent = "center";
    statusContainer.style.alignItems = "center";
    statusContainer.style.height = "300px";
}


// Function to ping the server for each station
async function fetchStatuses(customHour = null) {
    const hour = customHour !== null ? customHour : getCurrentHourGMT8();
    const nextHour = (hour + 1) % 24;

    // Ensure it's within operating hours
    if (hour >= 5 && (hour !== 5 || currentMinute >= 30) && hour < 24) {
        for (const station of stations) {
            const payload = {
                station: station,
                day: day,
                hour: nextHour
            };

            try {
                const response = await fetch(BACKEND_URL, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });

                const result = await response.json();
                displayStatusCard(station, nextHour, result.probabilities);
            } catch (error) {
                console.error("Error fetching data for station:", station, error);
            }
        }
    }
}

// Function to create and display a card with images
function displayStatusCard(station, hour, probabilities) {
    const prefix = station.slice(0, 2).toLowerCase();
    const lineName = lineNames[prefix] || "Unknown Line";
    let lineContainer = document.getElementById(`line-${prefix}`);

    // Format the hour to display in 12-hour format with AM/PM
    const formattedHour = formatHour(hour);

    // Create a new line group if it doesn't exist
    if (!lineContainer) {
        const statusContainer = document.getElementById("status-container");

        lineContainer = document.createElement("div");
        lineContainer.id = `line-${prefix}`;
        lineContainer.className = "line-group";
        lineContainer.innerHTML = `
            <div class="line-header">
                <img src="/static/images/MRT_Crowd_Images/Lines/${prefix}.jpg" alt="${lineName} Line" onerror="this.src='/static/images/MRT_Crowd_Images/Lines/Default.jpg';">
            </div>
            <h2>${lineName} Line</h2>
            <div class="search-container">
                <input type="text" class="lineSearchBar" id="searchBar-${prefix}" placeholder="Search for a station..." onkeyup="filterStations('${prefix}')">
            </div>
            <div class="line-cards"></div>
        `;
        statusContainer.appendChild(lineContainer);
    }

    // Prevent duplicate station cards
    const lineCards = lineContainer.querySelector(".line-cards");
    if (lineCards.querySelector(`#station-${station}`)) {
        return;
    }

    const card = document.createElement("div");
    card.id = `station-${station}`;
    card.className = "card";

    const probabilitiesArray = [
        { label: "Low", value: probabilities.l },
        { label: "Medium", value: probabilities.m },
        { label: "High", value: probabilities.h }
    ];

    probabilitiesArray.sort((a, b) => b.value - a.value);
    const primaryPrediction = probabilitiesArray[0];
    const secondaryPrediction = probabilitiesArray[1];

    card.innerHTML = `
        <img src="/static/images/MRT_Crowd_Images/Stations/${station}.jpg" alt="${station}" onerror="this.src='/static/images/MRT_Crowd_Images/Stations/Default.jpg';">
        <h2>Station: ${station.toUpperCase()}</h2>
        <div class="cardParagraph">
            <p class="hour">Upcoming Hour: ${formattedHour}</p>
            <p>Predicted Crowd Level: <strong>${primaryPrediction.label}</strong></p>
            <p>With chance of <strong>${secondaryPrediction.label}</strong>: ${secondaryPrediction.value.toFixed(2)}%</p>
        </div>
    `;

    lineCards.appendChild(card);
}

// Function to toggle the modal visibility
function toggleModal() {
    const modal = document.getElementById("queryModal");
    if (modal.style.display === "flex") {
      modal.style.display = "none";
    } else {
      modal.style.display = "flex";
    }
  }
  
// Function to send a message to dialogflow
async function sendMessage() {
    const userInput = document.getElementById("userInput").value.trim();
    const chatContent = document.getElementById("chatContent");

    if (!userInput) {
        alert("Please enter a message.");
        return;
    }

    // Append user input to chatContent
    const userMessage = document.createElement("div");
    userMessage.textContent = "You: " + userInput;
    userMessage.style.color = "blue";
    chatContent.appendChild(userMessage);

    document.getElementById("userInput").value = "";

    chatContent.scrollTop = chatContent.scrollHeight;

    const webhookUrl = "http://127.0.0.1:5000/webhook";

    const payload = {
        queryInput: {
        text: {
            text: userInput,
            languageCode: "en",
        },
        },
    };

    try {
        const response = await fetch(webhookUrl, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
        });

        const data = await response.json();

        // Append AI response to chatContent
        const aiMessage = document.createElement("div");
        aiMessage.textContent = "AI: " + data.fulfillmentText;
        aiMessage.style.color = "green";
        chatContent.appendChild(aiMessage);

        // Scroll to the bottom of chat content
        chatContent.scrollTop = chatContent.scrollHeight;
    } catch (error) {
        console.error("Error:", error);

    // Append fallback error message to chatContent
    const errorMessage = document.createElement("div");
    errorMessage.textContent = "AI: I didn't get that. Can you repeat?";
    errorMessage.style.color = "red";
    chatContent.appendChild(errorMessage);

    // Scroll to the bottom of chat content
    chatContent.scrollTop = chatContent.scrollHeight;
    }
}
  
  
// Function to automatically ping every hour
function startAutoPing() {
    fetchStatuses();

    // Schedule the next fetch to be at the start of the next hour
    const msUntilNextHour = (60 - now.getMinutes()) * 60000 - now.getSeconds() * 1000;
    setTimeout(() => {
        fetchStatuses();
        setInterval(fetchStatuses, 60 * 60 * 1000);
    }, msUntilNextHour);
}

    // Path to the PDF image of the railways
    const pdfPath = '/static/images/MRT_Crowd_Images/SM_Eng_(Ver101224)_PC.pdf';

    // PDF.js settings
    let pdfDoc = null,
        pageNum = 1,
        scale = 2.6, 
        canvas = document.getElementById('pdf-render'),
        ctx = canvas.getContext('2d'),
        pdfContainer = document.getElementById('pdf-container');

    // Load the PDF
    pdfjsLib.getDocument(pdfPath).promise.then(pdf => {
        pdfDoc = pdf;
        renderPage(pageNum);
    });

// function for centering the pdf image
function centerView() {
const container = document.getElementById('pdf-container');
const canvas = document.getElementById('pdf-render');

// Calculate the center scroll position
const scrollLeft = (canvas.width - container.clientWidth) / 2;
const scrollTop = (canvas.height - container.clientHeight) / 2;

// Set the scroll position
container.scrollLeft = scrollLeft > 0 ? scrollLeft : 0; // Ensure it doesn't scroll to negative values
container.scrollTop = scrollTop > 0 ? scrollTop : 0;
}

// Render the specified page of the PDF onto the canvas element
function renderPage(num) {
    pdfDoc.getPage(num).then(page => {
        // Get the dimensions of the PDF page at the current scale
        const viewport = page.getViewport({ scale });

        // Set the canvas dimensions to match the PDF page
        canvas.width = viewport.width;
        canvas.height = viewport.height;

        // Prepare the rendering context for the canvas
        const renderContext = {
            canvasContext: ctx,
            viewport: viewport
        };

        // Render the PDF page on the canvas and center it
        page.render(renderContext).promise.then(() => {
            centerView(); 
        });
    });
}

// Reset Zoom
function resetZoom() {
    scale = 2.6; 
    renderPage(pageNum);
}

let isDragging = false; // Track if the user is dragging
let startX = 0; // Mouse X position at the start of drag
let startY = 0; // Mouse Y position at the start of drag
let scrollLeft = 0; // Container's scrollLeft position at the start of drag
let scrollTop = 0; // Container's scrollTop position at the start of drag

// Mouse Down Event: Start Dragging
pdfContainer.addEventListener('mousedown', (event) => {
    isDragging = true;
    pdfContainer.classList.add('dragging');
    startX = event.clientX; // Get initial mouse X position
    startY = event.clientY; // Get initial mouse Y position
    scrollLeft = pdfContainer.scrollLeft; // Record initial scroll position
    scrollTop = pdfContainer.scrollTop; // Record initial scroll position
});

// Mouse Move Event: Drag the PDF
pdfContainer.addEventListener('mousemove', (event) => {
    if (!isDragging) return; // If not dragging, exit

    // Prevent default behavior to avoid text selection
    event.preventDefault();

    // Calculate the distance moved
    const dx = event.clientX - startX; // Horizontal movement
    const dy = event.clientY - startY; // Vertical movement

    // Scroll the container
    pdfContainer.scrollLeft = scrollLeft - dx;
    pdfContainer.scrollTop = scrollTop - dy;
});

// Mouse Up Event: Stop Dragging
pdfContainer.addEventListener('mouseup', () => {
    isDragging = false;
    pdfContainer.classList.remove('dragging');
});

// Mouse Leave Event: Stop Dragging if the mouse leaves the container
pdfContainer.addEventListener('mouseleave', () => {
    isDragging = false;
    pdfContainer.classList.remove('dragging');
});
let debounceTimer = null;

pdfContainer.addEventListener('wheel', (event) => {
    event.preventDefault();

    // Update the zoom scale
    if (event.deltaY < 0) {
        // Scroll up to zoom in
        scale += 0.2;
    } else if (event.deltaY > 0) {
        // Scroll down to zoom out
        if (scale > 0.4) scale -= 0.2; // Minimum zoom level
    }

    // Clear the existing debounce timer
    if (debounceTimer) clearTimeout(debounceTimer);

    // Set a new debounce timer to delay rendering
    debounceTimer = setTimeout(() => {
        renderPage(pageNum);
    }, 100); // Adjust delay time (100ms) as needed for smoother experience
});

// Zoom on Scroll
pdfContainer.addEventListener('wheel', (event) => {
    event.preventDefault();

    if (event.deltaY < 0) {
        // Scroll up to zoom in
        scale += 0.2;
    } else if (event.deltaY > 0) {
        // Scroll down to zoom out
        if (scale > 0.4) scale -= 0.2; // Minimum zoom level
    }

    renderPage(pageNum);
});

function filterLines() {
    const query = document.getElementById('searchBar').value.toLowerCase(); 
    const allLineGroups = document.querySelectorAll('.line-group'); 

    // Loop through each line group and toggle visibility 
    allLineGroups.forEach(lineGroup => {
    const lineName = lineGroup.querySelector('h2').textContent.toLowerCase(); 
    if (lineName.includes(query)) {
        lineGroup.style.display = 'block'; 
    } else {
        lineGroup.style.display = 'none'; 
    }
    });
}

// Function to filter stations within a specific line
function filterStations(linePrefix) {
    const query = document.getElementById(`searchBar-${linePrefix}`).value.toLowerCase();
    const lineCards = document.querySelector(`#line-${linePrefix} .line-cards`);
    const stationCards = lineCards.querySelectorAll(".card");

    // Loop through each station card and toggle visibility
    stationCards.forEach(card => {
        const stationName = card.querySelector("h2").textContent.toLowerCase();
        card.style.display = stationName.includes(query) ? "block" : "none";
    });
}


document.addEventListener("DOMContentLoaded", () => {
    checkOperatingHours();
    startAutoPing();
});

// List of all stations
const stations = [
    'BP1', 'BP2', 'BP3', 'BP4', 'BP5', 'BP6', 'BP7', 'BP8', 'BP9', 'BP10', 'BP11', 'BP12', 'BP13',
    'CC1', 'CC2', 'CC3', 'CC4', 'CC5', 'CC6', 'CC7', 'CC8', 'CC9', 'CC10', 'CC11', 'CC12', 'CC13', 'CC14', 'CC15', 'CC16', 'CC17', 'CC19', 'CC20', 'CC21', 'CC22', 'CC23', 'CC24', 'CC25', 'CC26', 'CC27', 'CC28', 'CC29',
    'CE1', 'CE2',
    'CG1', 'CG2',
    'DT1', 'DT2', 'DT3', 'DT5', 'DT6', 'DT7', 'DT8', 'DT9', 'DT10', 'DT11', 'DT12', 'DT13', 'DT14', 'DT15', 'DT16', 'DT17', 'DT18', 'DT19', 'DT20', 'DT21', 'DT22', 'DT23', 'DT24', 'DT25', 'DT26', 'DT27', 'DT28', 'DT29', 'DT30', 'DT31', 'DT32', 'DT33', 'DT34', 'DT35',
    'EW1', 'EW2', 'EW3', 'EW4', 'EW5', 'EW6', 'EW7', 'EW8', 'EW9', 'EW10', 'EW11', 'EW12', 'EW13', 'EW14', 'EW15', 'EW16', 'EW17', 'EW18', 'EW19', 'EW20', 'EW21', 'EW22', 'EW23', 'EW24', 'EW25', 'EW26', 'EW27', 'EW28', 'EW29', 'EW30', 'EW31', 'EW32', 'EW33',
    'NE1', 'NE3', 'NE4', 'NE5', 'NE6', 'NE7', 'NE8', 'NE9', 'NE10', 'NE11', 'NE12', 'NE13', 'NE14', 'NE15', 'NE16', 'NE17',
    'NS1', 'NS2', 'NS3', 'NS4', 'NS5', 'NS7', 'NS8', 'NS9', 'NS10', 'NS11', 'NS12', 'NS13', 'NS14', 'NS15', 'NS16', 'NS17', 'NS18', 'NS19', 'NS20', 'NS21', 'NS22', 'NS23', 'NS24', 'NS25', 'NS26', 'NS27', 'NS28',
    'PE1', 'PE2', 'PE3', 'PE4', 'PE5', 'PE6', 'PE7',
    'PTC',
    'PW1', 'PW3', 'PW4', 'PW5', 'PW6', 'PW7',
    'SE1', 'SE2', 'SE3', 'SE4', 'SE5',
    'STC',
    'SW1', 'SW2', 'SW3', 'SW4', 'SW5', 'SW6', 'SW7', 'SW8',
    'TE1', 'TE2', 'TE3', 'TE4', 'TE5', 'TE6', 'TE7', 'TE8', 'TE9', 'TE11', 'TE12', 'TE13', 'TE14', 'TE15', 'TE16', 'TE17', 'TE18', 'TE19', 'TE20', 'TE22', 'TE23', 'TE24', 'TE25', 'TE26', 'TE27', 'TE28', 'TE29', 
];

const lineNames = {
    ns: "North South",
    ew: "East West",
    dt: "Downtown",
    cc: "Circle",
    bp: "Bukit Panjang LRT",
    ce: "Circle Extension",
    cg: "Changi Airport",
    ne: "North East",
    te: "Thompson East-Coast",
    pe: "Punggol East LRT",
    pw: "Punggol West LRT",
    pt: "Punggol LRT Terminal",
    se: "Sengkang East LRT",
    sw: "Sengkang West LRT",
    st: "Sengkang LRT Terminal"
    };