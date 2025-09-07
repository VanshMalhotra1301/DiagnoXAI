document.addEventListener('DOMContentLoaded', () => {
    // --- Initialize Choices.js for the searchable dropdown ---
    const symptomsSelect = document.getElementById('symptoms');
    const choices = new Choices(symptomsSelect, {
        removeItemButton: true,
        placeholder: true,
        placeholderValue: 'Type to search for symptoms...',
        searchPlaceholderValue: 'Type here...',
        allowHTML: true
    });

    // --- Get references to other DOM elements ---
    const form = document.getElementById('symptom-form');
    const predictBtn = document.getElementById('predict-btn');
    const btnText = document.querySelector('.btn-text');
    const spinner = document.querySelector('.spinner');
    const resultContainer = document.getElementById('result-container');
    
    // --- Listen for the form's submit event ---
    form.addEventListener('submit', (event) => {
        event.preventDefault(); // Prevent default page reload
        btnText.textContent = 'Analyzing...';
        spinner.classList.remove('hidden');
        predictBtn.disabled = true;
        resultContainer.classList.add('hidden');

        fetch('/predict', { method: 'POST', body: new FormData(form) })
            .then(response => response.json())
            .then(data => { displayResults(data); })
            .catch(error => {
                console.error('Error:', error);
                displayResults({ error: 'An unexpected error occurred. Please try again.' });
            })
            .finally(() => {
                btnText.textContent = 'Analyze Symptoms';
                spinner.classList.add('hidden');
                predictBtn.disabled = false;
            });
    });

    // --- Function to dynamically create and show the result card ---
    function displayResults(data) {
        let content = '';
        if (data.error) {
            content = `<div class="glass-card result-card error">...</div>`; // Simplified for brevity
        } else {
            content = `
                <div class="glass-card result-card">
                    <div class="result-header">
                        <i class="ph ph-first-aid-kit"></i>
                        <span>AI Analysis Complete</span>
                    </div>
                    <div class="result-body">
                        <h3 class="result-title">Potential Condition</h3>
                        <h2>${data.prediction}</h2>
                        <p class="suggestion-title">Recommended Action</p>
                        <p>${data.suggestion}</p>
                    </div>
                </div>`;
        }
        resultContainer.innerHTML = content;
        resultContainer.classList.remove('hidden');
        // Re-apply the mousemove listener to the newly created result card
        addSpotlightEffect(resultContainer.querySelector('.glass-card'));
    }

    // --- Function to add the spotlight effect to a card ---
    function addSpotlightEffect(card) {
        if (!card) return;
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            card.style.setProperty('--mouse-x', `${x}px`);
            card.style.setProperty('--mouse-y', `${y}px`);
        });
    }

    // --- Apply spotlight effect to all initial glass cards ---
    document.querySelectorAll('.glass-card').forEach(card => {
        addSpotlightEffect(card);
    });
});