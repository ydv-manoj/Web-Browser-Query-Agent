// Web Query Agent Frontend JavaScript

// Configuration - can be overridden by window.API_CONFIG
const API_BASE_URL = window.API_CONFIG?.baseUrl || 'http://localhost:8000';
const API_TIMEOUT = window.API_CONFIG?.timeout || 60000;

// Global state
let currentQuery = '';
let searchInProgress = false;

// DOM elements
const searchForm = document.getElementById('searchForm');
const queryInput = document.getElementById('queryInput');
const useCacheInput = document.getElementById('useCacheInput');
const loadingSection = document.getElementById('loadingSection');
const loadingStatus = document.getElementById('loadingStatus');
const progressBarFill = document.getElementById('progressBarFill');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Set up event listeners
    searchForm.addEventListener('submit', handleSearch);
    queryInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSearch(e);
        }
    });
    
    // Focus on the search input
    queryInput.focus();
    
    // Load initial stats
    loadStats();
    
    console.log('🚀 Initializing Web Query Agent...');
    console.log('🔗 API Base URL:', API_BASE_URL);
    console.log('✅ Web Query Agent initialized');
}

// Search functionality
async function handleSearch(e) {
    e.preventDefault();
    
    if (searchInProgress) {
        showToast('Search already in progress', 'warning');
        return;
    }
    
    const query = queryInput.value.trim();
    if (!query) {
        showToast('Please enter a search query', 'warning');
        queryInput.focus();
        return;
    }
    
    currentQuery = query;
    searchInProgress = true;
    
    // Update UI
    hideAllSections();
    showLoadingSection();
    updateSearchButton(true);
    
    try {
        // Start progress simulation
        simulateProgress();
        
        // Make API request
        const response = await fetch(`${API_BASE_URL}/api/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                use_cache: useCacheInput.checked
            })
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.detail || 'Search failed');
        }
        
        if (result.success) {
            displayResults(result);
            showToast('Search completed successfully', 'success');
        } else {
            displayError(result.error || result.message);
        }
        
    } catch (error) {
        console.error('Search error:', error);
        displayError(error.message);
        showToast('Search failed: ' + error.message, 'error');
    } finally {
        searchInProgress = false;
        updateSearchButton(false);
        hideLoadingSection();
    }
}

function simulateProgress() {
    const steps = [
        { progress: 20, message: 'Validating query...' },
        { progress: 40, message: 'Checking cache...' },
        { progress: 60, message: 'Searching the web...' },
        { progress: 80, message: 'Scraping content...' },
        { progress: 95, message: 'Generating summary...' }
    ];
    
    let currentStep = 0;
    
    const updateProgress = () => {
        if (currentStep < steps.length && searchInProgress) {
            const step = steps[currentStep];
            setProgress(step.progress);
            setLoadingStatus(step.message);
            currentStep++;
            setTimeout(updateProgress, 1000 + Math.random() * 1000);
        }
    };
    
    updateProgress();
}

function setProgress(percentage) {
    if (progressBarFill) {
        progressBarFill.style.width = percentage + '%';
    }
}

function setLoadingStatus(message) {
    if (loadingStatus) {
        loadingStatus.textContent = message;
    }
}

function updateSearchButton(loading) {
    const submitBtn = searchForm.querySelector('button[type="submit"]');
    if (submitBtn) {
        if (loading) {
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Searching...';
        } else {
            submitBtn.disabled = false;
            submitBtn.innerHTML = '<i class="fas fa-search"></i> Search';
        }
    }
}

// Display functions
function displayResults(result) {
    hideAllSections();
    
    // Update result header
    const resultTitle = document.getElementById('resultTitle');
    const executionTime = document.getElementById('executionTime');
    const cacheStatus = document.getElementById('cacheStatus');
    
    if (resultTitle) {
        if (result.similar_query_used) {
            resultTitle.textContent = `Results for "${result.query}" (Similar Query Used)`;
        } else {
            resultTitle.textContent = `Results for "${result.query}"`;
        }
    }
    
    // if (executionTime) {
    //     executionTime.innerHTML = `<i class="fas fa-clock"></i> ${result.execution_time.toFixed(2)}s`;
    // }
    
    if (cacheStatus) {
        const isCacheHit = result.cache_hit || result.similar_query_used;
        cacheStatus.innerHTML = isCacheHit 
            ? '<i class="fas fa-database"></i> Cached'
            : '<i class="fas fa-globe"></i> Fresh';
    }
    
    if (result.result && result.result.summary) {
        const summary = result.result.summary;
        
        // Display summary
        const summaryContent = document.getElementById('summaryContent');
        if (summaryContent) {
            summaryContent.textContent = summary.summary || 'No summary available.';
        }
        
        // Display key points
        displayKeyPoints(summary.key_points || []);
        
        // Display sources
        displaySources(summary.sources || []);
        
        // Display scraping details
        displayScrapingDetails(result.result.scraped_content || []);
    }
    
    // Show similar query info
    if (result.similar_query_used) {
        showToast(
            `Used similar query: "${result.similar_query_used.original_query}" (${(result.similar_query_used.similarity_score * 100).toFixed(1)}% match)`,
            'info'
        );
    }
    
    resultsSection.classList.remove('hidden');
}

function displayKeyPoints(keyPoints) {
    const keyPointsList = document.getElementById('keyPointsList');
    if (!keyPointsList) return;
    
    keyPointsList.innerHTML = '';
    
    if (keyPoints.length === 0) {
        keyPointsList.innerHTML = '<li>No key points extracted.</li>';
        return;
    }
    
    keyPoints.forEach(point => {
        const li = document.createElement('li');
        li.textContent = point;
        keyPointsList.appendChild(li);
    });
}

function displaySources(sources) {
    const sourcesList = document.getElementById('sourcesList');
    if (!sourcesList) return;
    
    sourcesList.innerHTML = '';
    
    if (sources.length === 0) {
        sourcesList.innerHTML = '<div class="text-muted">No sources available.</div>';
        return;
    }
    
    sources.forEach((source, index) => {
        const sourceDiv = document.createElement('div');
        sourceDiv.className = 'source-item';
        
        sourceDiv.innerHTML = `
            <span class="source-icon">
                <i class="fas fa-external-link-alt"></i>
            </span>
            <a href="${source}" target="_blank" rel="noopener noreferrer">
                ${getDomainFromUrl(source)}
            </a>
        `;
        
        sourcesList.appendChild(sourceDiv);
    });
}

function displayScrapingDetails(scrapedContent) {
    const scrapingDetails = document.getElementById('scrapingDetails');
    if (!scrapingDetails) return;
    
    scrapingDetails.innerHTML = '';
    
    if (scrapedContent.length === 0) {
        scrapingDetails.innerHTML = '<div class="text-muted">No scraping details available.</div>';
        return;
    }
    
    const successCount = scrapedContent.filter(c => c.success).length;
    const totalCount = scrapedContent.length;
    
    const summaryDiv = document.createElement('div');
    summaryDiv.innerHTML = `
        <p><strong>Scraping Summary:</strong> ${successCount}/${totalCount} pages successfully scraped</p>
        <div class="mt-2">
    `;
    scrapingDetails.appendChild(summaryDiv);
    
    scrapedContent.forEach((content, index) => {
        const itemDiv = document.createElement('div');
        itemDiv.className = 'scraping-item';
        
        const statusClass = content.success ? 'status-success' : 'status-error';
        const statusIcon = content.success ? 'fa-check-circle' : 'fa-times-circle';
        const statusText = content.success ? 'Success' : 'Failed';
        
        itemDiv.innerHTML = `
            <div class="scraping-title">${content.title || 'Untitled'}</div>
            <div class="scraping-meta">
                <span class="${statusClass}">
                    <i class="fas ${statusIcon}"></i> ${statusText}
                </span>
                ${content.success ? `<span>${content.content_length} chars</span>` : ''}
                ${!content.success && content.error_message ? `<span title="${content.error_message}">Error</span>` : ''}
            </div>
        `;
        
        scrapingDetails.appendChild(itemDiv);
    });
}

function displayError(message) {
    hideAllSections();
    errorMessage.textContent = message || 'An unexpected error occurred';
    errorSection.classList.remove('hidden');
}

function hideError() {
    errorSection.classList.add('hidden');
    queryInput.focus();
}

function hideAllSections() {
    loadingSection.classList.add('hidden');
    resultsSection.classList.add('hidden');
    errorSection.classList.add('hidden');
}

function showLoadingSection() {
    loadingSection.classList.remove('hidden');
    setProgress(10);
    setLoadingStatus('Starting search...');
}

function hideLoadingSection() {
    loadingSection.classList.add('hidden');
    setProgress(0);
}

// Modal functions
function showModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
        
        // Close on backdrop click
        modal.addEventListener('click', function(e) {
            if (e.target === modal) {
                hideModal(modalId);
            }
        });
        
        // Close on Escape key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                hideModal(modalId);
            }
        });
    }
}

function hideModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.add('hidden');
        document.body.style.overflow = '';
    }
}

// Stats functionality
async function showStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/stats`);
        const stats = await response.json();
        
        if (!response.ok) {
            throw new Error(stats.detail || 'Failed to load stats');
        }
        
        // Update stats display
        document.getElementById('totalQueries').textContent = stats.total_queries;
        document.getElementById('cacheHitRate').textContent = stats.cache_hit_rate.toFixed(1) + '%';
        document.getElementById('avgExecutionTime').textContent = stats.average_execution_time.toFixed(2) + 's';
        document.getElementById('totalPages').textContent = stats.total_pages_scraped;
        document.getElementById('vectorEntries').textContent = stats.total_entries;
        document.getElementById('dbSize').textContent = stats.cache_size_mb.toFixed(2) + ' MB';
        
        // Detailed stats
        document.getElementById('validQueries').textContent = stats.valid_queries;
        document.getElementById('invalidQueries').textContent = stats.invalid_queries;
        document.getElementById('cacheHits').textContent = stats.cache_hits;
        document.getElementById('cacheMisses').textContent = stats.cache_misses;
        document.getElementById('lastUpdated').textContent = formatDateTime(stats.last_updated);
        
        showModal('statsModal');
        
    } catch (error) {
        console.error('Stats error:', error);
        showToast('Failed to load statistics: ' + error.message, 'error');
    }
}

function hideStats() {
    hideModal('statsModal');
}

async function loadStats() {
    // Load stats silently for health check
    try {
        const response = await fetch(`${API_BASE_URL}/api/stats`);
        if (response.ok) {
            console.log('✅ API connection successful');
        }
    } catch (error) {
        console.warn('⚠️ API connection failed:', error);
    }
}

// History functionality
async function showHistory() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/history?limit=50`);
        const historyData = await response.json();
        
        if (!response.ok) {
            throw new Error(historyData.detail || 'Failed to load history');
        }
        
        displayHistoryList(historyData.queries);
        showModal('historyModal');
        
    } catch (error) {
        console.error('History error:', error);
        showToast('Failed to load history: ' + error.message, 'error');
    }
}

function hideHistory() {
    hideModal('historyModal');
}

function displayHistoryList(queries) {
    const historyList = document.getElementById('historyList');
    if (!historyList) return;
    
    historyList.innerHTML = '';
    
    if (queries.length === 0) {
        historyList.innerHTML = '<div class="text-center text-muted">No queries found.</div>';
        return;
    }
    
    queries.forEach(query => {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        
        const statusClass = query.status === 'valid' ? 'status-valid' : 'status-invalid';
        
        historyItem.innerHTML = `
            <div class="history-query">${query.query}</div>
            <div class="history-meta">
                <div>
                    <span class="status-badge ${statusClass}">${query.status}</span>
                    <span>Confidence: ${(query.confidence * 100).toFixed(1)}%</span>
                    <span>Pages: ${query.pages_scraped}</span>
                    <span>${formatDateTime(query.created_at)}</span>
                </div>
                <div class="history-actions">
                    <button class="btn btn-sm btn-primary" onclick="rerunQuery('${query.query}')">
                        <i class="fas fa-redo"></i> Rerun
                    </button>
                    <button class="btn btn-sm btn-danger" onclick="deleteQuery('${query.id}')">
                        <i class="fas fa-trash"></i> Delete
                    </button>
                </div>
            </div>
        `;
        
        historyList.appendChild(historyItem);
    });
}

async function searchHistory() {
    const searchTerm = document.getElementById('historySearchInput').value.trim();
    
    try {
        const url = searchTerm 
            ? `${API_BASE_URL}/api/history?limit=50&search=${encodeURIComponent(searchTerm)}`
            : `${API_BASE_URL}/api/history?limit=50`;
            
        const response = await fetch(url);
        const historyData = await response.json();
        
        if (!response.ok) {
            throw new Error(historyData.detail || 'Failed to search history');
        }
        
        displayHistoryList(historyData.queries);
        
    } catch (error) {
        console.error('History search error:', error);
        showToast('Failed to search history: ' + error.message, 'error');
    }
}

async function refreshHistory() {
    await showHistory();
    showToast('History refreshed', 'success');
}

function rerunQuery(query) {
    hideHistory();
    queryInput.value = query;
    queryInput.focus();
    queryInput.select();
}

async function deleteQuery(queryId) {
    if (!confirm('Are you sure you want to delete this query?')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/history/${queryId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to delete query');
        }
        
        await refreshHistory();
        showToast('Query deleted successfully', 'success');
        
    } catch (error) {
        console.error('Delete error:', error);
        showToast('Failed to delete query: ' + error.message, 'error');
    }
}

// Validation functionality
async function validateQuery() {
    const query = queryInput.value.trim();
    if (!query) {
        showToast('Please enter a query to validate', 'warning');
        queryInput.focus();
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/validate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: query })
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.detail || 'Validation failed');
        }
        
        displayValidationResults(result);
        showModal('validationModal');
        
    } catch (error) {
        console.error('Validation error:', error);
        showToast('Failed to validate query: ' + error.message, 'error');
    }
}

function displayValidationResults(result) {
    const validationResults = document.getElementById('validationResults');
    if (!validationResults) return;
    
    const statusClass = result.status === 'valid' ? 'validation-valid' : 'validation-invalid';
    const statusIcon = result.status === 'valid' ? 'fa-check-circle' : 'fa-times-circle';
    
    let html = `
        <div class="validation-status ${statusClass}">
            <i class="fas ${statusIcon}"></i>
            Query is ${result.status}
        </div>
        <div class="validation-confidence">
            Confidence: ${(result.confidence * 100).toFixed(1)}%
        </div>
        <div class="validation-reason">
            ${result.reason}
        </div>
    `;
    
    if (result.suggested_improvements && result.suggested_improvements.length > 0) {
        html += `
            <div class="validation-suggestions">
                <h4>Suggestions for improvement:</h4>
                <ul>
                    ${result.suggested_improvements.map(suggestion => `<li>${suggestion}</li>`).join('')}
                </ul>
            </div>
        `;
    }
    
    validationResults.innerHTML = html;
}

function hideValidation() {
    hideModal('validationModal');
}

// Cache management
async function clearCache() {
    if (!confirm('Are you sure you want to clear all cached data? This action cannot be undone.')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/cache`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to clear cache');
        }
        
        showToast('Cache cleared successfully', 'success');
        
        // Refresh stats if modal is open
        const statsModal = document.getElementById('statsModal');
        if (statsModal && !statsModal.classList.contains('hidden')) {
            await showStats();
        }
        
    } catch (error) {
        console.error('Clear cache error:', error);
        showToast('Failed to clear cache: ' + error.message, 'error');
    }
}

// Utility functions
function toggleDetails() {
    const detailsCard = document.querySelector('.details-card');
    const detailsContent = document.getElementById('scrapingDetails');
    
    if (detailsCard && detailsContent) {
        const isExpanded = !detailsContent.classList.contains('hidden');
        
        if (isExpanded) {
            detailsContent.classList.add('hidden');
            detailsCard.classList.remove('expanded');
        } else {
            detailsContent.classList.remove('hidden');
            detailsCard.classList.add('expanded');
        }
    }
}

function getDomainFromUrl(url) {
    try {
        const domain = new URL(url).hostname;
        return domain.replace('www.', '');
    } catch {
        return url;
    }
}

function formatDateTime(dateString) {
    try {
        const date = new Date(dateString);
        return date.toLocaleString();
    } catch {
        return dateString;
    }
}

// Toast notifications
function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toastContainer');
    if (!toastContainer) return;
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icon = getToastIcon(type);
    toast.innerHTML = `
        <i class="fas ${icon}"></i>
        <span>${message}</span>
    `;
    
    toastContainer.appendChild(toast);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    }, 5000);
    
    // Remove on click
    toast.addEventListener('click', () => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    });
}

function getToastIcon(type) {
    switch (type) {
        case 'success': return 'fa-check-circle';
        case 'error': return 'fa-times-circle';
        case 'warning': return 'fa-exclamation-triangle';
        case 'info': 
        default: return 'fa-info-circle';
    }
}

// Export functions for global access
window.showStats = showStats;
window.hideStats = hideStats;
window.showHistory = showHistory;
window.hideHistory = hideHistory;
window.refreshHistory = refreshHistory;
window.searchHistory = searchHistory;
window.rerunQuery = rerunQuery;
window.deleteQuery = deleteQuery;
window.validateQuery = validateQuery;
window.hideValidation = hideValidation;
window.clearCache = clearCache;
window.toggleDetails = toggleDetails;
window.hideError = hideError;