const scanBtn = document.getElementById('scanBtn');
const btnText = document.querySelector('.btn-text');
const loader = document.querySelector('.loader');
const commentInput = document.getElementById('commentInput');
const resultsSection = document.getElementById('resultsSection');
const scoreDisplay = document.getElementById('scoreDisplay');
const verdictText = document.getElementById('verdictText');
const scoreRing = document.getElementById('scoreRing');
const ringProgress = document.getElementById('ringProgress');
const errorBox = document.getElementById('errorBox');

const API_URL = 'http://127.0.0.1:8000/predict';

scanBtn.addEventListener('click', async () => {
    const text = commentInput.value.trim();
    if (!text) return;

    // Enter UI Loading State
    scanBtn.disabled = true;
    btnText.classList.add('hidden');
    loader.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    errorBox.classList.remove('show');
    
    // Reset previously applied toxicity classes and score
    resultsSection.classList.remove('is-toxic');
    scoreDisplay.textContent = '0%';
    
    // Reset ring animation position
    ringProgress.style.strokeDashoffset = 283;

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text })
        });

        if (!response.ok) throw new Error('API Error');

        const data = await response.json();
        
        // Calculate dynamic properties
        const percentage = Math.round(data.toxicity * 100);
        const circumference = 283; // For SVG ring length
        const offset = circumference - (percentage / 100) * circumference;

        // Artificial delay purely to allow the user to see the loading state/glassmorphism blur before result pop
        setTimeout(() => {
            // Restore button
            scanBtn.disabled = false;
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
            
            // Make results visible
            resultsSection.classList.remove('hidden');
            
            // Adjust visual themes based on outcome
            resultsSection.classList.remove('is-toxic');
            if (data.label === 'toxic') {
                resultsSection.classList.add('is-toxic');
                verdictText.textContent = 'Harmful Content Detected';
            } else {
                verdictText.textContent = 'Safe / Clean';
            }

            // Animate SVG Ring
            ringProgress.style.strokeDashoffset = offset;
            
            // Animate percentage text counting up
            let currentNum = 0;
            const animateCount = setInterval(() => {
                if (currentNum >= percentage) {
                    clearInterval(animateCount);
                    scoreDisplay.textContent = `${percentage}%`;
                } else {
                    currentNum++;
                    scoreDisplay.textContent = `${currentNum}%`;
                }
            }, 10);
            
        }, 600);
        
    } catch (err) {
        scanBtn.disabled = false;
        btnText.classList.remove('hidden');
        loader.classList.add('hidden');
        
        errorBox.classList.add('show');
        setTimeout(() => {
            errorBox.classList.remove('show');
        }, 3000);
    }
});
