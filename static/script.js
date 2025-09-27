document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultBox = document.getElementById('result-box');
    const salaryDisplay = document.getElementById('predicted-salary-display');
    const predictBtn = document.getElementById('predict-btn');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // 1. Collect Data - Collect all values using element IDs
        const jobTitle = document.getElementById('job-title').value;
        const educationLevel = document.getElementById('education-level').value; 
        const experience = document.getElementById('experience').value;
        const age = document.getElementById('age').value; 
        const gender = document.getElementById('gender').value; 

        if (!jobTitle || !educationLevel || !experience || !age || !gender) {
            alert("Please fill in all fields.");
            return;
        }

        // === CRITICAL: Send JSON with the original, uncleaned keys ===
        // The Flask server handles cleaning these keys: "Years of Experience" -> "years_of_experience"
        const data = {
            "Job Title": jobTitle,
            "Years of Experience": parseFloat(experience), 
            "Education Level": educationLevel,
            "Age": parseInt(age), 
            "Gender": gender
        };
        
        // 2. Set UI for loading state
        resultBox.classList.remove('hidden');
        salaryDisplay.textContent = 'Calculating...';
        predictBtn.disabled = true;
        predictBtn.textContent = 'Predicting...';

        // 3. API Call
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });

            // Handle successful response
            if (response.ok) {
                const result = await response.json();
                const salary = result.predicted_salary;

                // 4. Update UI with result
                salaryDisplay.textContent = formatCurrency(salary);
                resultBox.style.opacity = '0';
                setTimeout(() => {
                    resultBox.style.opacity = '1';
                }, 50);

            } else {
                // Handle API error
                const error = await response.json();
                salaryDisplay.textContent = `Error: ${error.error || 'Prediction failed'}`;
                resultBox.style.background = 'linear-gradient(45deg, #e74c3c, #c0392b)';
            }

        } catch (error) {
            // Handle network/fetch error
            console.error('Fetch error:', error);
            salaryDisplay.textContent = 'Network Error. Check console.';
            resultBox.style.background = 'linear-gradient(45deg, #e74c3c, #c0392b)';
        } finally {
            // 5. Reset button state
            predictBtn.disabled = false;
            predictBtn.textContent = 'Predict Salary';
            if (resultBox.style.background.includes('#e74c3c')) {
                 resultBox.style.background = 'linear-gradient(45deg, #4a90e2, #50e3c2)'; 
            }
        }
    });

    function formatCurrency(number) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 0
        }).format(number);
    }
});