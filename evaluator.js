// CRS Arena Evaluation Logic
// Ported from eval.py to JavaScript

const TURN_ASPECTS = ["relevance", "interestingness"];
const DIALOGUE_ASPECTS = ["understanding", "task_completion", "interest_arousal", "efficiency", "dialogue_overall"];
const DATASET_ORDER = ["redial", "opendialkg"];

let goldData = null;

// Statistical functions
function mean(arr) {
    return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function standardDeviation(arr) {
    const avg = mean(arr);
    const squareDiffs = arr.map(value => Math.pow(value - avg, 2));
    return Math.sqrt(mean(squareDiffs));
}

function pearsonCorrelation(x, y) {
    if (x.length !== y.length || x.length === 0) {
        return NaN;
    }
    
    const n = x.length;
    const meanX = mean(x);
    const meanY = mean(y);
    
    let numerator = 0;
    let sumXSquared = 0;
    let sumYSquared = 0;
    
    for (let i = 0; i < n; i++) {
        const diffX = x[i] - meanX;
        const diffY = y[i] - meanY;
        numerator += diffX * diffY;
        sumXSquared += diffX * diffX;
        sumYSquared += diffY * diffY;
    }
    
    const denominator = Math.sqrt(sumXSquared * sumYSquared);
    
    if (denominator === 0) {
        return NaN;
    }
    
    return numerator / denominator;
}

function spearmanCorrelation(x, y) {
    if (x.length !== y.length || x.length === 0) {
        return NaN;
    }
    
    // Create ranks for x and y
    const rankX = getRanks(x);
    const rankY = getRanks(y);
    
    // Calculate Pearson correlation on ranks
    return pearsonCorrelation(rankX, rankY);
}

function getRanks(arr) {
    // Create array of [value, originalIndex]
    const indexed = arr.map((val, idx) => ({ val, idx }));
    
    // Sort by value
    indexed.sort((a, b) => a.val - b.val);
    
    // Assign ranks (handle ties by averaging)
    const ranks = new Array(arr.length);
    let i = 0;
    
    while (i < indexed.length) {
        let j = i;
        // Find all tied values
        while (j < indexed.length && indexed[j].val === indexed[i].val) {
            j++;
        }
        
        // Average rank for tied values
        const rank = (i + j + 1) / 2;
        
        for (let k = i; k < j; k++) {
            ranks[indexed[k].idx] = rank;
        }
        
        i = j;
    }
    
    return ranks;
}

// Data loading functions
async function loadGoldData() {
    try {
        const response = await fetch('crs_arena_eval/crs_arena_eval.json');
        if (!response.ok) {
            throw new Error(`Failed to load gold data: ${response.statusText}`);
        }
        const data = await response.json();
        return parseGoldData(data);
    } catch (error) {
        throw new Error(`Error loading gold data: ${error.message}`);
    }
}

function parseGoldData(goldDataJson) {
    const turnGold = new Map();
    const dialGold = new Map();
    
    for (const dialog of goldDataJson) {
        const convId = dialog.conv_id;
        dialGold.set(convId, dialog.dial_level_aggregated || {});
        
        for (const turn of dialog.dialogue || []) {
            if (turn.role !== "ASST") {
                continue;
            }
            const key = `${convId}:${turn.turn_ind}`;
            turnGold.set(key, turn.turn_level_aggregated || {});
        }
    }
    
    return { turnGold, dialGold };
}

function parseRunData(runDataJson) {
    const turnPreds = new Map();
    const dialPreds = new Map();
    
    for (const dialog of runDataJson) {
        const convId = dialog.conv_id;
        const turnList = dialog.turns || [];
        
        for (const turn of turnList) {
            const turnInd = parseInt(turn.turn_ind);
            const key = `${convId}:${turnInd}`;
            const turnLevelPred = {};
            
            for (const aspect of TURN_ASPECTS) {
                if (turn.turn_level_pred && aspect in turn.turn_level_pred) {
                    turnLevelPred[aspect] = parseFloat(turn.turn_level_pred[aspect]);
                }
            }
            
            turnPreds.set(key, turnLevelPred);
        }
        
        const dialLevelPred = {};
        for (const aspect of DIALOGUE_ASPECTS) {
            if (dialog.dial_level_pred && aspect in dialog.dial_level_pred) {
                dialLevelPred[aspect] = parseFloat(dialog.dial_level_pred[aspect]);
            }
        }
        dialPreds.set(convId, dialLevelPred);
    }
    
    return { turnPreds, dialPreds };
}

function datasetFromConvId(convId) {
    const parts = convId.split('_');
    if (parts.length < 2) {
        throw new Error(`Unexpected conv_id format: ${convId}`);
    }
    return parts[1];
}

function systemFromConvId(convId) {
    const parts = convId.split('_');
    if (parts.length < 2) {
        throw new Error(`Unexpected conv_id format: ${convId}`);
    }
    return `${parts[0]}_${parts[1]}`;
}

function computePerSystemSpearman(turnPreds, turnGold, dialPreds, dialGold) {
    const aspects = [...TURN_ASPECTS, ...DIALOGUE_ASPECTS];
    const bySystem = new Map();

    function ensureSystem(systemId) {
        if (!bySystem.has(systemId)) {
            const aspectBuckets = {};
            for (const aspect of aspects) {
                aspectBuckets[aspect] = { pred: [], gold: [] };
            }
            bySystem.set(systemId, aspectBuckets);
        }
    }

    for (const [key, goldAspects] of turnGold) {
        const [convId] = key.split(':');
        if (!turnPreds.has(key)) {
            continue;
        }
        const predAspects = turnPreds.get(key);
        const systemId = systemFromConvId(convId);
        ensureSystem(systemId);

        for (const aspect of TURN_ASPECTS) {
            if (!(aspect in goldAspects) || !(aspect in predAspects)) {
                continue;
            }
            const bucket = bySystem.get(systemId)[aspect];
            bucket.pred.push(predAspects[aspect]);
            bucket.gold.push(goldAspects[aspect]);
        }
    }

    for (const [convId, goldAspects] of dialGold) {
        if (!dialPreds.has(convId)) {
            continue;
        }
        const predAspects = dialPreds.get(convId);
        const systemId = systemFromConvId(convId);
        ensureSystem(systemId);

        for (const aspect of DIALOGUE_ASPECTS) {
            if (!(aspect in goldAspects) || !(aspect in predAspects)) {
                continue;
            }
            const bucket = bySystem.get(systemId)[aspect];
            bucket.pred.push(predAspects[aspect]);
            bucket.gold.push(goldAspects[aspect]);
        }
    }

    const systemLabels = Array.from(bySystem.keys()).sort();
    const values = systemLabels.map(systemId => {
        const aspectBuckets = bySystem.get(systemId);
        return aspects.map(aspect => {
            const bucket = aspectBuckets[aspect];
            if (!bucket || bucket.pred.length === 0) {
                return NaN;
            }
            return spearmanCorrelation(bucket.pred, bucket.gold);
        });
    });

    return {
        labels: systemLabels,
        aspects,
        values
    };
}

function computeMetrics(records) {
    const byDataset = {};
    
    for (const dataset of DATASET_ORDER) {
        byDataset[dataset] = { pred: [], gold: [] };
    }
    
    for (const { dataset, pred, gold } of records) {
        if (byDataset[dataset]) {
            byDataset[dataset].pred.push(pred);
            byDataset[dataset].gold.push(gold);
        }
    }
    
    const metrics = {};
    
    for (const dataset of DATASET_ORDER) {
        const preds = byDataset[dataset].pred;
        const golds = byDataset[dataset].gold;
        
        if (preds.length === 0) {
            metrics[dataset] = { pearson: NaN, spearman: NaN };
            continue;
        }
        
        const pearson = pearsonCorrelation(preds, golds);
        const spearman = spearmanCorrelation(preds, golds);
        
        metrics[dataset] = { pearson, spearman };
    }
    
    return metrics;
}

function evaluateTurnLevel(turnPreds, turnGold) {
    const results = {};
    
    for (const aspect of TURN_ASPECTS) {
        const records = [];
        
        for (const [key, goldAspects] of turnGold) {
            if (!(aspect in goldAspects)) {
                continue;
            }
            
            if (!turnPreds.has(key)) {
                continue;
            }
            
            const predAspects = turnPreds.get(key);
            if (!(aspect in predAspects)) {
                continue;
            }
            
            const convId = key.split(':')[0];
            const dataset = datasetFromConvId(convId);
            
            records.push({
                dataset,
                pred: predAspects[aspect],
                gold: goldAspects[aspect]
            });
        }
        
        results[aspect] = computeMetrics(records);
    }
    
    return results;
}

function evaluateDialogueLevel(dialPreds, dialGold) {
    const results = {};
    
    for (const aspect of DIALOGUE_ASPECTS) {
        const records = [];
        
        for (const [convId, goldAspects] of dialGold) {
            if (!(aspect in goldAspects)) {
                continue;
            }
            
            if (!dialPreds.has(convId)) {
                continue;
            }
            
            const predAspects = dialPreds.get(convId);
            if (!(aspect in predAspects)) {
                continue;
            }
            
            const dataset = datasetFromConvId(convId);
            
            records.push({
                dataset,
                pred: predAspects[aspect],
                gold: goldAspects[aspect]
            });
        }
        
        results[aspect] = computeMetrics(records);
    }
    
    return results;
}

function formatNumber(num) {
    if (isNaN(num)) {
        return 0; // Use 0 for N/A values in charts
    }
    return parseFloat(num.toFixed(3));
}

function formatSpearmanCell(num) {

    if (isNaN(num)) {
        return "N/A";
    }
    return num.toFixed(3);
}
function extractSystemName(convId) {
    // Extract system_dataset from conv_id (e.g., "barcor_redial_..." -> "barcor_redial")
    const parts = convId.split('_');
    if (parts.length >= 2) {
        return `${parts[0]}_${parts[1]}`;
    }
    return parts[0];
}

function computeSystemMetrics(turnPreds, dialPreds, turnGold, dialGold) {
    const systemData = {};
    
    // Collect data per system
    for (const [key, goldAspects] of turnGold) {
        if (!turnPreds.has(key)) continue;
        
        const convId = key.split(':')[0];
        const system = extractSystemName(convId);
        const predAspects = turnPreds.get(key);
        
        if (!systemData[system]) {
            systemData[system] = {};
        }
        
        for (const aspect of TURN_ASPECTS) {
            if (!(aspect in goldAspects) || !(aspect in predAspects)) continue;
            
            if (!systemData[system][aspect]) {
                systemData[system][aspect] = { pred: [], gold: [] };
            }
            
            systemData[system][aspect].pred.push(predAspects[aspect]);
            systemData[system][aspect].gold.push(goldAspects[aspect]);
        }
    }
    
    for (const [convId, goldAspects] of dialGold) {
        if (!dialPreds.has(convId)) continue;
        
        const system = extractSystemName(convId);
        const predAspects = dialPreds.get(convId);
        
        if (!systemData[system]) {
            systemData[system] = {};
        }
        
        for (const aspect of DIALOGUE_ASPECTS) {
            if (!(aspect in goldAspects) || !(aspect in predAspects)) continue;
            
            if (!systemData[system][aspect]) {
                systemData[system][aspect] = { pred: [], gold: [] };
            }
            
            systemData[system][aspect].pred.push(predAspects[aspect]);
            systemData[system][aspect].gold.push(goldAspects[aspect]);
        }
    }
    
    // Compute Spearman correlation for each system and aspect
    const systemMetrics = {};
    
    for (const system in systemData) {
        systemMetrics[system] = {};
        
        for (const aspect in systemData[system]) {
            const preds = systemData[system][aspect].pred;
            const golds = systemData[system][aspect].gold;
            
            if (preds.length > 0) {
                systemMetrics[system][aspect] = spearmanCorrelation(preds, golds);
            } else {
                systemMetrics[system][aspect] = NaN;
            }
        }
    }
    
    return systemMetrics;
}

function displaySystemTable(systemMetrics) {
    const tableBody = document.getElementById('systemTableBody');
    tableBody.innerHTML = '';
    
    const allAspects = [...TURN_ASPECTS, ...DIALOGUE_ASPECTS];
    const aspectLabels = {
        'relevance': 'Relevance',
        'interestingness': 'Interestingness',
        'understanding': 'Understanding',
        'task_completion': 'Task Completion',
        'interest_arousal': 'Interest Arousal',
        'efficiency': 'Efficiency',
        'dialogue_overall': 'Dialogue Overall'
    };
    
    // Sort systems alphabetically
    const systems = Object.keys(systemMetrics).sort();
    
    // Compute min/max for each aspect (column) for color scaling
    const aspectRanges = {};
    for (const aspect of allAspects) {
        const values = systems
            .map(sys => systemMetrics[sys][aspect])
            .filter(v => !isNaN(v));
        
        if (values.length > 0) {
            aspectRanges[aspect] = {
                min: Math.min(...values),
                max: Math.max(...values)
            };
        }
    }
    
    // Helper function to get color based on value within range
    function getColorForValue(value, min, max) {
        if (isNaN(value)) return 'transparent';
        
        // Normalize value to 0-1 range
        const range = max - min;
        const normalized = range > 0 ? (value - min) / range : 0.5;
        
        // Color gradient: red (low) -> yellow (mid) -> green (high)
        let r, g, b;
        if (normalized < 0.5) {
            // Red to Yellow
            r = 255;
            g = Math.round(255 * (normalized * 2));
            b = 0;
        } else {
            // Yellow to Green
            r = Math.round(255 * (1 - (normalized - 0.5) * 2));
            g = 255;
            b = 0;
        }
        
        // Return semi-transparent color for better readability
        return `rgba(${r}, ${g}, ${b}, 0.3)`;
    }
    
    for (const system of systems) {
        const row = document.createElement('tr');
        
        // System name cell
        const systemCell = document.createElement('td');
        systemCell.textContent = system;
        row.appendChild(systemCell);
        
        // Aspect correlation cells with color coding
        for (const aspect of allAspects) {
            const cell = document.createElement('td');
            const value = systemMetrics[system][aspect];
            
            if (formatNumber(value) === 0) {
                cell.textContent = 'N/A';
            } else {
                cell.textContent = value.toFixed(3);
                
                // Apply color gradient based on column (aspect) range
                if (aspectRanges[aspect]) {
                    const bgColor = getColorForValue(
                        value,
                        aspectRanges[aspect].min,
                        aspectRanges[aspect].max
                    );
                    cell.style.backgroundColor = bgColor;
                }
            }
            
            row.appendChild(cell);
        }
        
        tableBody.appendChild(row);
    }
}


function computeTurnBasedCorrelations(turnPreds, turnGold) {
    // Group predictions and gold by turn index
    const turnData = {};
    
    for (const [key, goldAspects] of turnGold) {
        if (!turnPreds.has(key)) continue;
        
        const turnInd = parseInt(key.split(':')[1]);
        const predAspects = turnPreds.get(key);
        
        if (!turnData[turnInd]) {
            turnData[turnInd] = {
                relevance: { pred: [], gold: [] },
                interestingness: { pred: [], gold: [] }
            };
        }
        
        // Collect turn-level aspects
        for (const aspect of TURN_ASPECTS) {
            if (aspect in goldAspects && aspect in predAspects) {
                turnData[turnInd][aspect].pred.push(predAspects[aspect]);
                turnData[turnInd][aspect].gold.push(goldAspects[aspect]);
            }
        }
    }
    
    // Compute Spearman correlation for each turn and each aspect
    const turnCorrelations = {};
    const turnNumbers = Object.keys(turnData).map(Number).sort((a, b) => a - b);
    
    for (const aspect of TURN_ASPECTS) {
        turnCorrelations[aspect] = [];
        
        for (const turnInd of turnNumbers) {
            const preds = turnData[turnInd][aspect].pred;
            const golds = turnData[turnInd][aspect].gold;
            
            if (preds.length >= 3) { // Need at least 3 points for correlation
                const corr = spearmanCorrelation(preds, golds);
                turnCorrelations[aspect].push({ turn: turnInd, correlation: corr });
            }
        }
    }
    
    return turnCorrelations;
}

let turnAnalysisChartInstance = null;

function displayTurnAnalysisChart(turnCorrelations) {
    // Destroy existing chart if it exists
    if (turnAnalysisChartInstance) {
        turnAnalysisChartInstance.destroy();
    }
    
    const canvas = document.getElementById('turnAnalysisChart');
    if (!canvas) {
        console.warn('Turn analysis chart canvas not found');
        return;
    }
    
    // Prepare data
    const relevanceData = turnCorrelations.relevance || [];
    const interestingnessData = turnCorrelations.interestingness || [];
    
    // Get all turn numbers (x-axis), filtered to max turn 15
    const allTurns = new Set([
        ...relevanceData.map(d => d.turn),
        ...interestingnessData.map(d => d.turn)
    ]);
    const turnLabels = Array.from(allTurns)
        .filter(turn => turn <= 15)
        .sort((a, b) => a - b);
    
    // Create datasets
    const datasets = [];
    
    if (relevanceData.length > 0) {
        datasets.push({
            label: 'Relevance',
            data: turnLabels.map(turn => {
                const point = relevanceData.find(d => d.turn === turn);
                return point ? point.correlation : null;
            }),
            borderColor: 'rgba(102, 126, 234, 1)',
            backgroundColor: 'rgba(102, 126, 234, 0.1)',
            borderWidth: 2,
            tension: 0,
            pointRadius: 4,
            pointHoverRadius: 6
        });
    }
    
    if (interestingnessData.length > 0) {
        datasets.push({
            label: 'Interestingness',
            data: turnLabels.map(turn => {
                const point = interestingnessData.find(d => d.turn === turn);
                return point ? point.correlation : null;
            }),
            borderColor: 'rgba(118, 75, 162, 1)',
            backgroundColor: 'rgba(118, 75, 162, 0.1)',
            borderWidth: 2,
            tension: 0,
            pointRadius: 4,
            pointHoverRadius: 6
        });
    }
    
    const ctx = canvas.getContext('2d');
    turnAnalysisChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: turnLabels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Turn Number',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    },
                    ticks: {
                        stepSize: 1
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Spearman Correlation',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    },
                    min: 0,
                    max: 1,
                    ticks: {
                        stepSize: 0.2
                    },
                    grid: {
                        color: function(context) {
                            if (context.tick.value === 0) {
                                return 'rgba(0, 0, 0, 0.3)';
                            }
                            return 'rgba(0, 0, 0, 0.1)';
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        font: {
                            size: 12
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + context.parsed.y.toFixed(3);
                        }
                    }
                }
            }
        }
    });
}
// Store chart instances to destroy them before creating new ones
let redialChartInstance = null;
let opendialKGChartInstance = null;

function displayResults(turnResults, dialogueResults, systemMetrics) {
    // Show results section first so canvas elements are visible
    document.getElementById('results').style.display = 'block';
    
    // Debug: Check if canvas elements exist
    const redialCanvas = document.getElementById('redialChart');
    const opendialKGCanvas = document.getElementById('opendialKGChart');
    
    console.log('ReDial canvas:', redialCanvas);
    console.log('OpenDialKG canvas:', opendialKGCanvas);
    
    if (!redialCanvas || !opendialKGCanvas) {
        console.error('Canvas elements not found!');
        showError('Chart canvas elements not found. Please refresh the page.');
        return;
    }
    
    // Destroy existing charts if they exist
    if (redialChartInstance) {
        redialChartInstance.destroy();
    }
    if (opendialKGChartInstance) {
        opendialKGChartInstance.destroy();
    }

    // Prepare data for all aspects (both turn-level and dialogue-level)
    const allAspects = [...TURN_ASPECTS, ...DIALOGUE_ASPECTS];
    const labels = allAspects.map(aspect => aspect.replace(/_/g, ' '));
    
    // Collect Pearson and Spearman data for both datasets
    const redialPearsonData = [];
    const redialSpearmanData = [];
    const opendialKGPearsonData = [];
    const opendialKGSpearmanData = [];
    
    // Process turn-level aspects
    for (const aspect of TURN_ASPECTS) {
        const stats = turnResults[aspect] || {};
        const redial = stats.redial || { pearson: NaN, spearman: NaN };
        const opendialkg = stats.opendialkg || { pearson: NaN, spearman: NaN };
        
        redialPearsonData.push(formatNumber(redial.pearson));
        redialSpearmanData.push(formatNumber(redial.spearman));
        opendialKGPearsonData.push(formatNumber(opendialkg.pearson));
        opendialKGSpearmanData.push(formatNumber(opendialkg.spearman));
    }
    
    // Process dialogue-level aspects
    for (const aspect of DIALOGUE_ASPECTS) {
        const stats = dialogueResults[aspect] || {};
        const redial = stats.redial || { pearson: NaN, spearman: NaN };
        const opendialkg = stats.opendialkg || { pearson: NaN, spearman: NaN };
        
        redialPearsonData.push(formatNumber(redial.pearson));
        redialSpearmanData.push(formatNumber(redial.spearman));
        opendialKGPearsonData.push(formatNumber(opendialkg.pearson));
        opendialKGSpearmanData.push(formatNumber(opendialkg.spearman));
    }
    
    // Create ReDial chart
    const redialCtx = document.getElementById('redialChart').getContext('2d');
    redialChartInstance = new Chart(redialCtx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Pearson',
                    data: redialPearsonData,
                    borderColor: 'rgba(102, 126, 234, 1)',
                    backgroundColor: 'rgba(102, 126, 234, 0.2)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(102, 126, 234, 1)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(102, 126, 234, 1)'
                },
                {
                    label: 'Spearman',
                    data: redialSpearmanData,
                    borderColor: 'rgba(118, 75, 162, 1)',
                    backgroundColor: 'rgba(118, 75, 162, 0.2)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(118, 75, 162, 1)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(118, 75, 162, 1)'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1,
                    min: 0,
                    ticks: {
                        stepSize: 0.2,
                        callback: function(value) {
                            return value.toFixed(1);
                        }
                    },
                    pointLabels: {
                        font: {
                            size: 12
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + context.parsed.r.toFixed(3);
                        }
                    }
                }
            }
        }
    });
    
    // Create OpenDialKG chart
    const opendialKGCtx = document.getElementById('opendialKGChart').getContext('2d');
    opendialKGChartInstance = new Chart(opendialKGCtx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Pearson',
                    data: opendialKGPearsonData,
                    borderColor: 'rgba(102, 126, 234, 1)',
                    backgroundColor: 'rgba(102, 126, 234, 0.2)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(102, 126, 234, 1)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(102, 126, 234, 1)'
                },
                {
                    label: 'Spearman',
                    data: opendialKGSpearmanData,
                    borderColor: 'rgba(118, 75, 162, 1)',
                    backgroundColor: 'rgba(118, 75, 162, 0.2)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(118, 75, 162, 1)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(118, 75, 162, 1)'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1,
                    min: 0,
                    ticks: {
                        stepSize: 0.2,
                        callback: function(value) {
                            return value.toFixed(1);
                        }
                    },
                    pointLabels: {
                        font: {
                            size: 12
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + context.parsed.r.toFixed(3);
                        }
                    }
                }
            }
        }
    });

    // Update per-system Spearman table
    displaySystemTable(systemMetrics);
}

function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    
    setTimeout(() => {
        errorDiv.style.display = 'none';
    }, 5000);
}

function showSuccess(message) {
    const successDiv = document.getElementById('success');
    successDiv.textContent = message;
    successDiv.style.display = 'block';
    
    setTimeout(() => {
        successDiv.style.display = 'none';
    }, 3000);
}

function showLoading(show) {
    document.getElementById('loading').style.display = show ? 'block' : 'none';
}

function showFileInfo(filename, size) {
    const fileInfoDiv = document.getElementById('fileInfo');
    fileInfoDiv.innerHTML = `
        <strong>File loaded:</strong> ${filename} (${(size / 1024).toFixed(2)} KB)
    `;
    fileInfoDiv.style.display = 'block';
}

async function processFile(file) {
    try {
        showLoading(true);
        document.getElementById('error').style.display = 'none';
        document.getElementById('results').style.display = 'none';
        
        // Load gold data if not already loaded
        if (!goldData) {
            goldData = await loadGoldData();
        }
        
        // Read and parse the uploaded file
        const fileContent = await file.text();
        const runDataJson = JSON.parse(fileContent);
        
        showFileInfo(file.name, file.size);
        
        // Parse run data
        const { turnPreds, dialPreds } = parseRunData(runDataJson);
        
        // Evaluate
        const turnResults = evaluateTurnLevel(turnPreds, goldData.turnGold);
        const dialogueResults = evaluateDialogueLevel(dialPreds, goldData.dialGold);
        const systemMetrics = computeSystemMetrics(turnPreds, dialPreds, goldData.turnGold, goldData.dialGold);
        const turnCorrelations = computeTurnBasedCorrelations(turnPreds, goldData.turnGold);
        
        // Display results
        displayResults(turnResults, dialogueResults, systemMetrics);
        displayTurnAnalysisChart(turnCorrelations);
        
        showSuccess('✅ Evaluation completed successfully!');
        showLoading(false);
        
    } catch (error) {
        console.error('Error processing file:', error);
        showError(`Error: ${error.message}`);
        showLoading(false);
    }
}

// Event handlers
document.getElementById('fileInput').addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        processFile(file);
    }
});

// Drag and drop functionality
const uploadSection = document.getElementById('uploadSection');

uploadSection.addEventListener('dragover', (event) => {
    event.preventDefault();
    uploadSection.classList.add('dragover');
});

uploadSection.addEventListener('dragleave', () => {
    uploadSection.classList.remove('dragover');
});

uploadSection.addEventListener('drop', (event) => {
    event.preventDefault();
    uploadSection.classList.remove('dragover');
    
    const file = event.dataTransfer.files[0];
    if (file && file.type === 'application/json') {
        processFile(file);
    } else {
        showError('Please upload a valid JSON file.');
    }
});

// Initialize: Load gold data on page load
window.addEventListener('DOMContentLoaded', async () => {
    try {
        goldData = await loadGoldData();
        console.log('Gold data loaded successfully');
    } catch (error) {
        console.error('Failed to load gold data:', error);
        showError('Failed to load evaluation data. Please check if crs_arena_eval.json is available.');
    }
});
