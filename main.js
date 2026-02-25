let session = null;

const statusEl = document.getElementById('status');
const loaderEl = document.getElementById('loader');
const form = document.getElementById('prediction-form');
const runButton = document.getElementById('run-button');
const resultContainer = document.getElementById('result-container');
const resultValue = document.getElementById('result-value');

// Helper to convert base64 to ArrayBuffer
function base64ToBuffer(base64) {
    const binString = atob(base64);
    const len = binString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binString.charCodeAt(i);
    }
    return bytes.buffer;
}

const sleep = (ms) => new Promise(res => setTimeout(res, ms));

async function initModel() {
    try {
        loaderEl.classList.remove('hidden');
        statusEl.textContent = 'Analyzing model...';

        if (typeof MODEL_BASE64 === 'undefined') {
            throw new Error('Model data missing. model_data.js not loaded.');
        }

        const modelBuffer = base64ToBuffer(MODEL_BASE64);

        const options = {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        };

        session = await ort.InferenceSession.create(modelBuffer, options);

        statusEl.textContent = 'Model Ready';
        statusEl.classList.add('ready');
        loaderEl.classList.add('hidden');
        runButton.disabled = false;

        console.log('--- DEBUG: Model Loaded ---');
        console.log('Input Names:', session.inputNames);
        console.log('Output Names:', session.outputNames);

        // Log all input metadata if available
        if (session.handler && session.handler.inputNamesToDataTypes) {
            console.log('Input Types:', Object.fromEntries(session.handler.inputNamesToDataTypes));
        }

    } catch (err) {
        console.error('Failed to load model:', err);
        statusEl.textContent = 'System Error';
        statusEl.classList.add('error');
        loaderEl.classList.add('hidden');
    }
}

// Scaling Parameters from scalerParams.json
// Feature Order: ["model", "condition", "cylinders", "fuel", "odometer", "transmission", "drive", "CarAge"]
const SCALING_MEAN = [177.62431826259606, 2.7129042932604515, 5.882453151618399, 0.9710882060961112, 121739.12073339625, 0.9238169941048865, 1.057431397301239, 17.380767958035616];
const SCALING_STD = [283.53371191795486, 0.7306282533024211, 1.6256224551274014, 0.3256881776561781, 206713.83456102648, 0.2652907000026549, 0.8860048265932386, 9.566076237447591];

// Populate Model Datalist
function populateModelList() {
    const list = document.getElementById('model-list');
    if (!list || typeof FREQ_MAP === 'undefined') return;

    // Sort models by frequency and take top 100 for performance (optional)
    // Here we just add all of them since it's a datalist
    Object.keys(FREQ_MAP).sort().forEach(modelName => {
        const option = document.createElement('option');
        option.value = modelName;
        list.appendChild(option);
    });
}

window.addEventListener('DOMContentLoaded', populateModelList);

form.onsubmit = async (e) => {
    e.preventDefault();
    if (!session) return;

    try {
        runButton.disabled = true;
        runButton.textContent = 'Calculating...';

        // 1. Collect Raw Inputs
        const modelName = document.getElementById('model').value.toLowerCase().trim();
        const modelFreq = FREQ_MAP[modelName] || 1; // Default to 1 if not found

        const rawInputs = {
            model: modelFreq,
            condition: parseFloat(document.getElementById('condition').value),
            cylinders: parseFloat(document.getElementById('cylinders').value),
            fuel: parseFloat(document.getElementById('fuel').value),
            odometer: parseFloat(document.getElementById('odometer').value),
            transmission: parseFloat(document.getElementById('transmission').value),
            drive: parseFloat(document.getElementById('drive').value),
            year: parseFloat(document.getElementById('year').value)
        };

        // 2. Initial Transformations
        if (rawInputs.odometer > 10000000) rawInputs.odometer = 10000000;
        const carAge = 2026 - rawInputs.year;

        // 3. Arrange in feature_names order:
        // ["model", "condition", "cylinders", "fuel", "odometer", "transmission", "drive", "CarAge"]
        const orderedInputs = [
            rawInputs.model,
            rawInputs.condition,
            rawInputs.cylinders,
            rawInputs.fuel,
            rawInputs.odometer,
            rawInputs.transmission,
            rawInputs.drive,
            carAge
        ];

        // 4. Apply Standard Scaling: (x - mean) / std
        const scaledInputs = orderedInputs.map((val, i) => (val - SCALING_MEAN[i]) / SCALING_STD[i]);

        console.log('--- DEBUG: Inference Start ---');
        console.log('Raw Values:', orderedInputs);
        console.log('Scaled Values:', scaledInputs);

        const inputName = session.inputNames[0];
        const data = new Float32Array(scaledInputs);
        const tensor = new ort.Tensor('float32', data, [1, 8]);

        const feeds = { [inputName]: tensor };
        const results = await session.run(feeds);

        console.log('Raw Results:', results);

        const outputName = session.outputNames[0];
        const output = results[outputName].data[0];

        // Format result
        resultValue.textContent = '$' + Number(output).toLocaleString(undefined, {
            minimumFractionDigits: 0,
            maximumFractionDigits: 0
        });

        resultContainer.classList.remove('hidden');
        console.log('Final Prediction:', output);

    } catch (err) {
        console.error('Inference error:', err);
        alert('Prediction failed. System reported: ' + err.message);
    } finally {
        runButton.disabled = false;
        runButton.textContent = 'Predict Price';
    }
};

initModel();
