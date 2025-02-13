const BACKEND_URL = "http://127.0.0.1:5000/predict/crowdlevel";

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
    for (const station of Object.keys(stations_dict)) {
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
    const stationName = stations_dict[station] || station;
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
        <h2>Station: ${stationName} (${station.toUpperCase()})</h2>
        <div class="cardParagraph">
            <p class="hour">Upcoming Hour: ${formattedHour}</p>
            <p>Predicted Crowd Level: <strong>${primaryPrediction.label}</strong></p>
            <p>With chance of <strong>${secondaryPrediction.label}</strong>: ${secondaryPrediction.value.toFixed(2)}%</p>
        </div>
    `;

    lineCards.appendChild(card);
}

function toggleModal() {
    const modal = document.getElementById("queryModal");

    if (modal.style.display === "flex") {
        modal.style.display = "none";
        document.removeEventListener("keydown", closeOnEsc); // Remove event listener when closing
    } else {
        modal.style.display = "flex";
        document.addEventListener("keydown", closeOnEsc); // Add event listener when opening
    }
}

function closeOnEsc(event) {
    if (event.key === "Escape") {
        const modal = document.getElementById("queryModal");
        modal.style.display = "none";
        document.removeEventListener("keydown", closeOnEsc); // Clean up listener
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

window.addEventListener('scroll', function() {
    let button = document.getElementById('scrollButton');
    if (window.scrollY > 100) {
        button.style.display = 'block';
    } else {
        button.style.display = 'none';
    }
});

function scrollToTop() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

document.addEventListener("DOMContentLoaded", () => {
    checkOperatingHours();
    startAutoPing();
});

// Dictionary of all stations
const stations_dict = {
    'BP1': 'Choa Chu Kang',
    'BP2': 'South View',
    'BP3': 'Keat Hong', 
    'BP4': 'Teck Whye',
    'BP5': 'Phoenix',
    'BP6': 'Bukit Panjang',
    'BP7': 'Petir',
    'BP8': 'Pending',
    'BP9': 'Bangkit',
    'BP10': 'Fajar',
    'BP11': 'Segar',
    'BP12': 'Jelapang',
    'BP13': 'Senja',
    'CC1': 'Dhoby Ghaut',
    'CC2': 'Bras Basah',
    'CC3': 'Esplanade',
    'CC4': 'Promenade',
    'CC5': 'Nicoll Highway',
    'CC6': 'Stadium',
    'CC7': 'Mountbatten',
    'CC8': 'Dakota',
    'CC9': 'Paya Lebar',
    'CC10': 'MacPherson',
    'CC11': 'Tai Seng',
    'CC12': 'Bartley',
    'CC13': 'Serangoon',
    'CC14': 'Lorong Chuan',
    'CC15': 'Bishan',
    'CC16': 'Marymount',
    'CC17': 'Caldecott',
    'CC19': 'Botanic Gardens',
    'CC20': 'Farrer Road',
    'CC21': 'Holland Village',
    'CC22': 'Buona Vista',
    'CC23': 'one-north',
    'CC24': 'Kent Ridge',
    'CC25': 'Haw Par Villa',
    'CC26': 'Pasir Panjang',
    'CC27': 'Labrador Park',
    'CC28': 'Telok Blangah',
    'CC29': 'HarbourFront',
    'CE1': 'Bayfront',
    'CE2': 'Marina Bay',
    'CG1': 'Expo',
    'CG2': 'Changi Airport',
    'DT1': 'Bukit Panjang',
    'DT2': 'Cashew',
    'DT3': 'Hillview',
    'DT5': 'Beauty World',
    'DT6': 'King Albert Park',
    'DT7': 'Sixth Avenue',
    'DT8': 'Tan Kah Kee',
    'DT9': 'Botanic Gardens',
    'DT10': 'Stevens',
    'DT11': 'Newton',
    'DT12': 'Little India',
    'DT13': 'Rochor',
    'DT14': 'Bugis',
    'DT15': 'Promenade',
    'DT16': 'Bayfront',
    'DT17': 'Downtown',
    'DT18': 'Telok Ayer',
    'DT19': 'Chinatown',
    'DT20': 'Fort Canning',
    'DT21': 'Bencoolen',
    'DT22': 'Jalan Besar',
    'DT23': 'Bendemeer',
    'DT24': 'Geylang Bahru',
    'DT25': 'Mattar',
    'DT26': 'MacPherson',
    'DT27': 'Ubi',
    'DT28': 'Kaki Bukit',
    'DT29': 'Bedok North',
    'DT30': 'Bedok Reservoir',
    'DT31': 'Tampines West',
    'DT32': 'Tampines',
    'DT33': 'Tampines East',
    'DT34': 'Upper Changi',
    'DT35': 'Expo',
    'EW1': 'Pasir Ris',
    'EW2': 'Tampines',
    'EW3': 'Simei',
    'EW4': 'Tanah Merah',
    'EW5': 'Bedok',
    'EW6': 'Kembangan',
    'EW7': 'Eunos',
    'EW8': 'Paya Lebar',
    'EW9': 'Aljunied',
    'EW10': 'Kallang',
    'EW11': 'Lavender',
    'EW12': 'Bugis',
    'EW13': 'City Hall',
    'EW14': 'Raffles Place',
    'EW15': 'Tanjong Pagar',
    'EW16': 'Outram Park',
    'EW17': 'Tiong Bahru',
    'EW18': 'Redhill',
    'EW19': 'Queenstown',
    'EW20': 'Commonwealth',
    'EW21': 'Buona Vista',
    'EW22': 'Dover',
    'EW23': 'Clementi',
    'EW24': 'Jurong East',
    'EW25': 'Chinese Garden',
    'EW26': 'Lakeside',
    'EW27': 'Boon Lay',
    'EW28': 'Pioneer',
    'EW29': 'Joo Koon',
    'EW30': 'Gul Circle',
    'EW31': 'Tuas Crescent',
    'EW32': 'Tuas West Road',
    'EW33': 'Tuas Link',
    'NE1': 'HarbourFront',
    'NE3': 'Outram Park',
    'NE4': 'Chinatown',
    'NE5': 'Clarke Quay',
    'NE6': 'Dhoby Ghaut',
    'NE7': 'Little India',
    'NE8': 'Farrer Park',
    'NE9': 'Boon Keng',
    'NE10': 'Potong Pasir',
    'NE11': 'Woodleigh',
    'NE12': 'Serangoon',
    'NE13': 'Kovan',
    'NE14': 'Hougang',
    'NE15': 'Buangkok',
    'NE16': 'Sengkang',
    'NE17': 'Punggol',
    'NS1': 'Jurong East',
    'NS2': 'Bukit Batok',
    'NS3': 'Bukit Gombak',
    'NS4': 'Choa Chu Kang',
    'NS5': 'Yew Tee',
    'NS7': 'Kranji',
    'NS8': 'Marsiling',
    'NS9': 'Woodlands',
    'NS10': 'Admiralty',
    'NS11': 'Sembawang',
    'NS12': 'Canberra',
    'NS13': 'Yishun',
    'NS14': 'Khatib',
    'NS15': 'Yio Chu Kang',
    'NS16': 'Ang Mo Kio',
    'NS17': 'Bishan',
    'NS18': 'Braddell',
    'NS19': 'Toa Payoh',
    'NS20': 'Novena',
    'NS21': 'Newton',
    'NS22': 'Orchard',
    'NS23': 'Somerset',
    'NS24': 'Dhoby Ghaut',
    'NS25': 'City Hall',
    'NS26': 'Raffles Place',
    'NS27': 'Marina Bay',
    'NS28': 'Marina South Pier',
    'PE1': 'Cove',
    'PE2': 'Meridian',
    'PE3': 'Coral Edge',
    'PE4': 'Riviera',
    'PE5': 'Kadaloor',
    'PE6': 'Oasis',
    'PE7': 'Damai',
    'PTC': 'Punggol', 
    'PW1': 'Sam Kee', 
    'PW3': 'Punggol Point',
    'PW4': 'Samudera', 
    'PW5': 'Nibong', 
    'PW6': 'Sumang', 
    'PW7': 'Soo Teck', 
    'SE1': 'Compassvale',
    'SE2': 'Rumbia', 
    'SE3': 'Bakau', 
    'SE4': 'Kangkar', 
    'SE5': 'Ranggung', 
    'STC': 'Sengkang', 
    'SW1': 'Cheng Lim', 
    'SW2': 'Farmway', 
    'SW3': 'Kupang', 
    'SW4': 'Thanggam', 
    'SW5': 'Fernvale', 
    'SW6': 'Layar', 
    'SW7': 'Tongkang', 
    'SW8': 'Renjong', 
    'TE1': 'Woodlands North', 
    'TE2': 'Woodlands', 
    'TE3': 'Woodlands South', 
    'TE4': 'Springleaf', 
    'TE5': 'Lentor', 
    'TE6': 'Mayflower', 
    'TE7': 'Bright Hill', 
    'TE8': 'Upper Thomson', 
    'TE9': 'Caldecott', 
    'TE11': 'Stevens', 
    'TE12': 'Napier', 
    'TE13': 'Orchard Boulevard', 
    'TE14': 'Orchard', 
    'TE15': 'Great World', 
    'TE16': 'Havelock', 
    'TE17': 'Outram Park', 
    'TE18': 'Maxwell', 
    'TE19': 'Shenton Way', 
    'TE20': 'Marina Bay', 
    'TE22': 'Gardens by the Bay', 
    'TE23': 'Tanjong Rhu', 
    'TE24': 'Katong Park', 
    'TE25': 'Tanjong Katong', 
    'TE26': 'Marine Parade', 
    'TE27': 'Marine Terrace', 
    'TE28': 'Siglap', 
    'TE29': 'Bayshore', 
}

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