// ================ AI日志异常检测模块 ================

// ========== 全局变量 ==========
let scoreGaugeChart = null;  // 分数仪表盘图表
let scoreChart = null;       // 批量检测分数趋势图表
let currentThreshold = 0.5;  // 当前阈值

// ========== 页面初始化 ==========
document.addEventListener('DOMContentLoaded', function() {
    // 初始化Lucide图标
    lucide.createIcons();
    
    // 初始化页面
    initPage();
});

function initPage() {
    // 获取模型状态
    fetchModelStatus();
    
    // 注册事件监听器
    document.getElementById('singleDetectBtn').addEventListener('click', detectSingleLog);
    document.getElementById('batchDetectBtn').addEventListener('click', detectBatchLogs);
    document.getElementById('refreshModelStatus').addEventListener('click', fetchModelStatus);
    document.getElementById('saveSettings').addEventListener('click', saveModelSettings);
    document.getElementById('thresholdRange').addEventListener('input', updateThresholdLabel);
    document.getElementById('themeToggle').addEventListener('click', toggleTheme);
    
    // 初始阈值设置
    document.getElementById('thresholdRange').value = currentThreshold;
    document.getElementById('thresholdLabel').textContent = currentThreshold;
}

// ========== 主题切换 ==========
function toggleTheme() {
    const htmlElement = document.documentElement;
    const isDark = htmlElement.getAttribute('data-bs-theme') === 'dark';
    const newTheme = isDark ? 'light' : 'dark';
    
    htmlElement.setAttribute('data-bs-theme', newTheme);
    
    // 更新图标
    document.querySelector('.theme-icon-light').classList.toggle('d-none');
    document.querySelector('.theme-icon-dark').classList.toggle('d-none');
    
    // 如果图表已初始化，更新图表主题
    updateChartsTheme(newTheme);
}

function updateChartsTheme(theme) {
    const textColor = theme === 'dark' ? '#f8f9fa' : '#212529';
    const gridColor = theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
    
    // 更新分数仪表盘图表
    if (scoreGaugeChart) {
        scoreGaugeChart.options.plugins.legend.labels.color = textColor;
        scoreGaugeChart.update();
    }
    
    // 更新分数趋势图表
    if (scoreChart) {
        scoreChart.options.scales.x.grid.color = gridColor;
        scoreChart.options.scales.y.grid.color = gridColor;
        scoreChart.options.scales.x.ticks.color = textColor;
        scoreChart.options.scales.y.ticks.color = textColor;
        scoreChart.options.plugins.legend.labels.color = textColor;
        scoreChart.update();
    }
}

// ========== 阈值设置 ==========
function updateThresholdLabel(e) {
    const value = e.target.value;
    document.getElementById('thresholdLabel').textContent = value;
}

// ========== API调用 ==========
// 获取模型状态
async function fetchModelStatus() {
    try {
        const response = await fetch('/ai/model/status');
        const data = await response.json();
        
        if (response.ok) {
            updateModelStatusUI(data);
        } else {
            showAlert('获取模型状态失败：' + (data.error || '未知错误'), 'danger');
        }
    } catch (error) {
        console.error('获取模型状态错误:', error);
        showAlert('无法连接到API服务', 'danger');
    }
}

// 更新模型状态UI
function updateModelStatusUI(data) {
    // 更新模型加载状态
    const modelLoadedEl = document.getElementById('modelLoaded');
    modelLoadedEl.textContent = data.model_loaded ? '已加载' : '未加载';
    modelLoadedEl.className = data.model_loaded ? 'badge bg-success' : 'badge bg-danger';
    
    // 更新设备信息
    const deviceEl = document.getElementById('modelDevice');
    deviceEl.textContent = data.device || 'CPU';
    deviceEl.className = data.device.includes('cuda') ? 'badge bg-info' : 'badge bg-secondary';
    
    // 更新KNN状态
    const knnEl = document.getElementById('modelKnn');
    knnEl.textContent = data.knn_enabled ? '已启用' : '未启用';
    knnEl.className = data.knn_enabled ? 'badge bg-success' : 'badge bg-secondary';
    
    // 更新嵌入向量数
    document.getElementById('embedCount').textContent = data.num_embeddings || '0';
    
    // 更新阈值
    if (data.threshold) {
        currentThreshold = data.threshold;
        document.getElementById('thresholdRange').value = currentThreshold;
        document.getElementById('thresholdLabel').textContent = currentThreshold;
    }
    
    // 更新全局KNN开关
    if ('knn_enabled' in data) {
        document.getElementById('globalKnnSwitch').checked = data.knn_enabled;
    }
}

// 保存模型设置
async function saveModelSettings() {
    // 获取阈值
    const threshold = parseFloat(document.getElementById('thresholdRange').value);
    
    // 获取KNN开关状态
    const knnEnabled = document.getElementById('globalKnnSwitch').checked;
    
    try {
        // 保存阈值
        const thresholdResponse = await fetch('/ai/model/threshold', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ threshold })
        });
        
        // 保存KNN状态
        const knnResponse = await fetch('/ai/knn/status', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ enabled: knnEnabled })
        });
        
        const thresholdData = await thresholdResponse.json();
        const knnData = await knnResponse.json();
        
        if (thresholdResponse.ok && knnResponse.ok) {
            showAlert('设置已保存', 'success');
            
            // 刷新模型状态
            fetchModelStatus();
        } else {
            const errorMsg = (thresholdData.error || '') + ' ' + (knnData.error || '');
            showAlert('保存设置失败：' + errorMsg, 'danger');
        }
    } catch (error) {
        console.error('保存设置错误:', error);
        showAlert('无法连接到API服务', 'danger');
    }
}

// ========== 单条日志检测 ==========
async function detectSingleLog() {
    // 获取日志文本
    const logText = document.getElementById('singleLogInput').value.trim();
    
    if (!logText) {
        showAlert('请输入日志内容', 'warning');
        return;
    }
    
    // 显示加载指示器
    document.getElementById('singleLoadingIndicator').style.display = 'block';
    document.getElementById('singleResultPanel').style.display = 'none';
    
    // 获取KNN设置
    const useKnn = document.getElementById('knnSwitch').checked;
    
    try {
        const response = await fetch('/ai/score_log', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                log: logText,
                use_knn: useKnn
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            updateSingleResultUI(data);
        } else {
            showAlert('检测失败：' + (data.error || '未知错误'), 'danger');
        }
    } catch (error) {
        console.error('检测错误:', error);
        showAlert('无法连接到API服务', 'danger');
    } finally {
        document.getElementById('singleLoadingIndicator').style.display = 'none';
    }
}

// 更新单条日志检测结果UI
function updateSingleResultUI(data) {
    // 处理分数
    let score = data.score;
    let scorePercent;
    
    // 检查score是否已经是百分比值（大于1）
    if (score > 1) {
        console.warn("API返回的异常分数大于1:", score);
        scorePercent = Math.round(score); // 假设API直接返回了百分比值
    } else {
        // 正常情况：score是0-1之间的值，转为百分比
        scorePercent = Math.round(score * 100);
    }
    
    // 确保分数在合理范围内
    if (scorePercent > 100) {
        console.warn("异常分数超过100%:", scorePercent, "使用原始值:", score);
        // 尝试规范化处理
        if (score > 100 && score < 10000) {
            // 可能是原始分数乘以了100
            scorePercent = Math.round(score / 100);
        } else {
            // 保持在100%以内
            scorePercent = 100;
        }
    }
    
    // 更新分数显示
    document.querySelector('.score-value').textContent = scorePercent + '%';
    
    // 更新异常标签
    const anomalyBadge = document.getElementById('anomalyBadge');
    if (data.is_anomaly) {
        anomalyBadge.textContent = '异常';
        anomalyBadge.className = 'badge bg-danger';
    } else {
        anomalyBadge.textContent = '正常';
        anomalyBadge.className = 'badge bg-success';
    }
    
    // 更新阈值显示
    document.getElementById('thresholdValue').textContent = data.threshold.toFixed(2);
    
    // 更新KNN状态
    document.getElementById('knnStatus').textContent = data.knn_used ? '使用' : '未使用';
    
    // 更新设备信息
    const deviceInfo = document.getElementById('deviceInfo');
    const modelDevice = document.getElementById('modelDevice').textContent;
    deviceInfo.textContent = modelDevice;
    
    // 更新结果解释
    let explanation = '该日志';
    // if (data.is_anomaly) {
    //     explanation += `被检测为异常，异常分数为 ${scorePercent}%，高于阈值 ${Math.round(data.threshold * 100)}%。`;
    // } else {
    //     explanation += `被检测为正常，异常分数为 ${scorePercent}%，低于阈值 ${Math.round(data.threshold * 100)}%。`;
    // }
    explanation += `异常分数为 ${scorePercent}%，阈值为 ${Math.round(data.threshold * 100)}%。`;
    document.getElementById('resultExplanation').textContent = explanation;
    
    // 创建或更新仪表盘图表
    // 确保分数在0-1范围内用于图表
    const normalizedScore = score > 1 ? Math.min(score / 100, 1) : score;
    createOrUpdateGaugeChart(normalizedScore);
    
    // 显示结果面板
    const resultPanel = document.getElementById('singleResultPanel');
    resultPanel.style.display = 'block';
    resultPanel.style.opacity = '0';
    
    // 淡入效果
    setTimeout(() => {
        resultPanel.style.opacity = '1';
    }, 10);
}

// 创建或更新仪表盘图表
function createOrUpdateGaugeChart(score) {
    const ctx = document.getElementById('scoreGauge').getContext('2d');
    
    // 配置仪表盘样式
    const getGradientColor = (value) => {
        // 绿色到红色的渐变
        const hue = ((1 - value) * 120).toString(10);
        return `hsl(${hue}, 100%, 50%)`;
    };
    
    const gaugeChartConfig = {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [score, 1 - score],
                backgroundColor: [
                    getGradientColor(score),
                    'rgba(200, 200, 200, 0.2)'
                ],
                borderWidth: 0
            }]
        },
        options: {
            circumference: 180,
            rotation: 270,
            cutout: '70%',
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                animateRotate: true,
                animateScale: true
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    enabled: false
                }
            }
        }
    };
    
    if (scoreGaugeChart) {
        scoreGaugeChart.data.datasets[0].data = [score, 1 - score];
        scoreGaugeChart.data.datasets[0].backgroundColor[0] = getGradientColor(score);
        scoreGaugeChart.update();
    } else {
        scoreGaugeChart = new Chart(ctx, gaugeChartConfig);
    }
}

// ========== 批量日志检测 ==========
async function detectBatchLogs() {
    // 获取日志文本
    const batchText = document.getElementById('batchLogInput').value.trim();
    
    if (!batchText) {
        showAlert('请输入批量日志内容', 'warning');
        return;
    }
    
    // 拆分为日志数组
    const logs = batchText.split('\n').filter(log => log.trim());
    
    if (logs.length === 0) {
        showAlert('没有有效的日志内容', 'warning');
        return;
    }
    
    // 显示加载指示器
    document.getElementById('batchLoadingIndicator').style.display = 'block';
    document.getElementById('batchResultPanel').style.display = 'none';
    
    // 获取窗口类型和步长
    const windowType = document.getElementById('windowTypeSelect').value;
    const stride = parseInt(document.getElementById('strideInput').value, 10) || 1;
    
    // 获取KNN设置
    const useKnn = document.getElementById('batchKnnSwitch').checked;
    
    try {
        const response = await fetch('/ai/detect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                logs: logs,
                window_type: windowType,
                stride: stride,
                use_knn: useKnn
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            updateBatchResultUI(data.result);
        } else {
            showAlert('批量检测失败：' + (data.error || '未知错误'), 'danger');
        }
    } catch (error) {
        console.error('批量检测错误:', error);
        showAlert('无法连接到API服务', 'danger');
    } finally {
        document.getElementById('batchLoadingIndicator').style.display = 'none';
    }
}

// 更新批量日志检测结果UI
function updateBatchResultUI(result) {
    // 更新统计信息
    document.getElementById('totalWindows').textContent = result.num_windows || 0;
    document.getElementById('anomalyWindows').textContent = result.num_anomaly_windows || 0;
    document.getElementById('anomalyRatio').textContent = `${Math.round((result.anomaly_ratio || 0) * 100)}%`;
    document.getElementById('avgScore').textContent = (result.avg_score || 0).toFixed(3);
    document.getElementById('maxScore').textContent = (result.max_score || 0).toFixed(3);
    
    // 保存窗口数据以便后续排序和过滤使用
    window.allWindowsData = result.windows || [];
    
    // 清空表格和列表
    const anomaliesTableBody = document.getElementById('anomaliesTableBody');
    const anomalyWindowsList = document.getElementById('anomalyWindowsList');
    anomaliesTableBody.innerHTML = '';
    anomalyWindowsList.innerHTML = '';
    
    if (window.allWindowsData && window.allWindowsData.length > 0) {
        // 默认按分数从高到低排序
        window.allWindowsData.sort((a, b) => b.score - a.score);
        
        // 构建表格和列表
        window.allWindowsData.forEach((window, index) => {
            // 为表格添加行
            const row = document.createElement('tr');
            row.className = 'log-row';
            row.setAttribute('data-score', window.score);
            
            // 计算分数样式
            let scoreClass = 'low-score';
            if (window.score > 0.7) {
                scoreClass = 'high-score';
            } else if (window.score >= 0.4) {
                scoreClass = 'medium-score';
            }
            
            // 计算分数百分比
            const scorePercent = Math.round(window.score * 100);
            
            // 行内容
            row.innerHTML = `
                <td>${index + 1}</td>
                <td>[${window.start_idx}-${window.end_idx}]</td>
                <td class="score-cell ${scoreClass}">${window.score.toFixed(4)}</td>
                <td>
                    <button class="btn btn-sm btn-outline-primary me-1" onclick="copyLogText(${index})">
                        <i data-lucide="clipboard"></i>
                    </button>
                    <button class="btn btn-sm btn-outline-secondary" onclick="toggleLogText(${index})">
                        <i data-lucide="chevrons-down" class="toggle-icon"></i>
                    </button>
                </td>
                <td>
                    <div class="log-text collapsed" id="log-text-${index}">${window.logs.join('<br>')}</div>
                </td>
            `;
            
            anomaliesTableBody.appendChild(row);
            
            // 为列表添加项
            const listItem = document.createElement('div');
            listItem.className = 'list-group-item';
            listItem.setAttribute('data-score', window.score);
            
            // 根据异常分数设置左边框颜色
            if (window.score > 0.7) {
                listItem.style.borderLeft = '4px solid var(--danger-color)';
            } else if (window.score >= 0.4) {
                listItem.style.borderLeft = '4px solid var(--warning-color)';
            } else {
                listItem.style.borderLeft = '4px solid var(--success-color)';
            }
            
            listItem.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <h6 class="mb-1">
                        <span class="score-indicator ${scoreClass > 0.7 ? 'high' : (scoreClass >= 0.4 ? 'medium' : 'low')}"></span>
                        窗口 #${index + 1} <small class="text-muted">[${window.start_idx}-${window.end_idx}]</small>
                    </h6>
                    <span class="badge ${scoreClass > 0.7 ? 'bg-danger' : (scoreClass >= 0.4 ? 'bg-warning' : 'bg-success')}">${scorePercent}%</span>
                </div>
                <div class="mt-2 small log-text">${window.logs.join('<br>')}</div>
                <div class="mt-2">
                    <button class="btn btn-sm btn-outline-primary" onclick="copyLogText(${index})">
                        <i data-lucide="clipboard" class="me-1"></i>复制
                    </button>
                </div>
            `;
            
            anomalyWindowsList.appendChild(listItem);
        });
        
        // 初始化图标
        lucide.createIcons();
        
        // 创建分数趋势图
        createScoreChart(window.allWindowsData);
    } else {
        anomaliesTableBody.innerHTML = '<tr><td colspan="5" class="text-center">没有检测到窗口数据</td></tr>';
        anomalyWindowsList.innerHTML = '<div class="list-group-item text-center">没有检测到窗口数据</div>';
    }
    
    // 显示结果面板
    const resultPanel = document.getElementById('batchResultPanel');
    resultPanel.style.display = 'block';
    resultPanel.style.opacity = '0';
    
    // 淡入效果
    setTimeout(() => {
        resultPanel.style.opacity = '1';
    }, 10);
}

// 创建分数趋势图
function createScoreChart(windows) {
    const ctx = document.getElementById('scoreChart').getContext('2d');
    
    // 提取数据
    const labels = windows.map(w => `窗口 ${w.window_idx}`);
    const scores = windows.map(w => w.score);
    
    // 获取当前主题
    const isDarkTheme = document.documentElement.getAttribute('data-bs-theme') === 'dark';
    const textColor = isDarkTheme ? '#f8f9fa' : '#212529';
    const gridColor = isDarkTheme ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
    
    // 配置图表
    const chartConfig = {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: '异常分数',
                data: scores,
                borderColor: 'rgb(13, 110, 253)',
                backgroundColor: 'rgba(13, 110, 253, 0.1)',
                borderWidth: 2,
                tension: 0.2,
                fill: true,
                pointBackgroundColor: (context) => {
                    const index = context.dataIndex;
                    return scores[index] > currentThreshold ? 'rgb(220, 53, 69)' : 'rgb(13, 110, 253)';
                },
                pointRadius: 4,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    grid: {
                        color: gridColor
                    },
                    ticks: {
                        color: textColor
                    }
                },
                x: {
                    grid: {
                        color: gridColor
                    },
                    ticks: {
                        color: textColor,
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: textColor
                    }
                },
                annotation: {
                    annotations: {
                        line1: {
                            type: 'line',
                            yMin: currentThreshold,
                            yMax: currentThreshold,
                            borderColor: 'rgb(220, 53, 69)',
                            borderWidth: 1,
                            borderDash: [5, 5]
                        }
                    }
                }
            }
        }
    };
    
    // 销毁旧图表
    if (scoreChart) {
        scoreChart.destroy();
    }
    
    // 创建新图表
    scoreChart = new Chart(ctx, chartConfig);
}

// ========== 工具函数 ==========
// 显示提示信息
function showAlert(message, type = 'info') {
    // 创建提示元素
    const alertEl = document.createElement('div');
    alertEl.className = `alert alert-${type} alert-dismissible fade show`;
    alertEl.setAttribute('role', 'alert');
    
    // 添加图标
    let icon = 'info';
    switch (type) {
        case 'success': icon = 'check-circle'; break;
        case 'danger': icon = 'alert-circle'; break;
        case 'warning': icon = 'alert-triangle'; break;
    }
    
    alertEl.innerHTML = `
        <i data-lucide="${icon}" class="me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="关闭"></button>
    `;
    
    // 插入到页面
    const container = document.querySelector('.container');
    container.insertBefore(alertEl, container.firstChild);
    
    // 初始化图标
    lucide.createIcons({
        icons: {
            'check-circle': true,
            'alert-circle': true,
            'alert-triangle': true,
            'info': true
        }
    });
    
    // 自动关闭
    setTimeout(() => {
        try {
            const bsAlert = new bootstrap.Alert(alertEl);
            bsAlert.close();
        } catch (e) {
            alertEl.remove();
        }
    }, 5000);
}

// 应用筛选条件
function applyFilters() {
    const searchText = document.getElementById('logSearchInput').value.toLowerCase();
    const scoreFilter = document.getElementById('scoreFilterSelect').value;
    
    // 筛选表格行
    const tableRows = document.querySelectorAll('#anomaliesTableBody .log-row');
    const tableVisibleCount = filterRows(tableRows, searchText, scoreFilter);
    
    // 筛选列表项
    const listItems = document.querySelectorAll('#anomalyWindowsList .list-group-item');
    const listVisibleCount = filterRows(listItems, searchText, scoreFilter);
    
    // 显示筛选结果信息
    const totalCount = window.allWindowsData ? window.allWindowsData.length : 0;
    const matchedCount = document.getElementById('tableViewTab').classList.contains('active') 
        ? tableVisibleCount : listVisibleCount;
    
    showAlert(`筛选结果: 显示 ${matchedCount} 条记录，共 ${totalCount} 条`, 'info');
}

// 筛选行
function filterRows(rows, searchText, scoreFilter) {
    let visibleCount = 0;
    
    rows.forEach(row => {
        const logText = row.querySelector('.log-text')?.textContent.toLowerCase() || 
                       row.textContent.toLowerCase();
        const score = parseFloat(row.getAttribute('data-score'));
        
        let showByScore = true;
        if (scoreFilter === 'high') {
            showByScore = score > 0.7;
        } else if (scoreFilter === 'medium') {
            showByScore = score >= 0.4 && score <= 0.7;
        } else if (scoreFilter === 'low') {
            showByScore = score < 0.4;
        }
        
        const showByText = searchText === '' || logText.includes(searchText);
        
        if (showByScore && showByText) {
            row.classList.remove('hidden');
            visibleCount++;
        } else {
            row.classList.add('hidden');
        }
    });
    
    return visibleCount;
}

// 表格排序
let currentSortColumn = 2; // 默认按分数排序
let currentSortDirection = -1; // -1表示降序，1表示升序

function sortTable(columnIndex, isNumeric = false) {
    if (!window.allWindowsData || window.allWindowsData.length === 0) return;
    
    // 如果点击当前排序列，则切换排序方向
    if (columnIndex === currentSortColumn) {
        currentSortDirection *= -1;
    } else {
        currentSortColumn = columnIndex;
        currentSortDirection = -1; // 新列默认降序
    }
    
    // 清除所有排序指示器
    const indicators = document.querySelectorAll('.sort-indicator');
    indicators.forEach(ind => ind.textContent = '');
    
    // 设置当前排序列的排序指示器
    const currentIndicator = document.querySelectorAll('th .sort-indicator')[columnIndex];
    currentIndicator.textContent = currentSortDirection === 1 ? ' ▲' : ' ▼';
    
    // 根据列执行排序
    switch (columnIndex) {
        case 0: // 序号
            window.allWindowsData.sort((a, b) => currentSortDirection);
            break;
        case 1: // 窗口位置
            window.allWindowsData.sort((a, b) => currentSortDirection * (a.start_idx - b.start_idx));
            break;
        case 2: // 异常分数
            window.allWindowsData.sort((a, b) => currentSortDirection * (a.score - b.score));
            break;
    }
    
    // 重新渲染表格
    updateTableWithCurrentData();
}

// 根据当前数据更新表格
function updateTableWithCurrentData() {
    const anomaliesTableBody = document.getElementById('anomaliesTableBody');
    const anomalyWindowsList = document.getElementById('anomalyWindowsList');
    anomaliesTableBody.innerHTML = '';
    anomalyWindowsList.innerHTML = '';
    
    if (window.allWindowsData && window.allWindowsData.length > 0) {
        window.allWindowsData.forEach((window, index) => {
            // 计算分数样式
            let scoreClass = 'low-score';
            if (window.score > 0.7) {
                scoreClass = 'high-score';
            } else if (window.score >= 0.4) {
                scoreClass = 'medium-score';
            }
            
            // 计算分数百分比
            const scorePercent = Math.round(window.score * 100);
            
            // 表格行
            const row = document.createElement('tr');
            row.className = 'log-row';
            row.setAttribute('data-score', window.score);
            
            row.innerHTML = `
                <td>${index + 1}</td>
                <td>[${window.start_idx}-${window.end_idx}]</td>
                <td class="score-cell ${scoreClass}">${window.score.toFixed(4)}</td>
                <td>
                    <button class="btn btn-sm btn-outline-primary me-1" onclick="copyLogText(${index})">
                        <i data-lucide="clipboard"></i>
                    </button>
                    <button class="btn btn-sm btn-outline-secondary" onclick="toggleLogText(${index})">
                        <i data-lucide="chevrons-down" class="toggle-icon"></i>
                    </button>
                </td>
                <td>
                    <div class="log-text collapsed" id="log-text-${index}">${window.logs.join('<br>')}</div>
                </td>
            `;
            
            anomaliesTableBody.appendChild(row);
            
            // 列表项
            const listItem = document.createElement('div');
            listItem.className = 'list-group-item';
            listItem.setAttribute('data-score', window.score);
            
            // 根据异常分数设置左边框颜色
            if (window.score > 0.7) {
                listItem.style.borderLeft = '4px solid var(--danger-color)';
            } else if (window.score >= 0.4) {
                listItem.style.borderLeft = '4px solid var(--warning-color)';
            } else {
                listItem.style.borderLeft = '4px solid var(--success-color)';
            }
            
            listItem.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <h6 class="mb-1">
                        <span class="score-indicator ${window.score > 0.7 ? 'high' : (window.score >= 0.4 ? 'medium' : 'low')}"></span>
                        窗口 #${index + 1} <small class="text-muted">[${window.start_idx}-${window.end_idx}]</small>
                    </h6>
                    <span class="badge ${window.score > 0.7 ? 'bg-danger' : (window.score >= 0.4 ? 'bg-warning' : 'bg-success')}">${scorePercent}%</span>
                </div>
                <div class="mt-2 small log-text">${window.logs.join('<br>')}</div>
                <div class="mt-2">
                    <button class="btn btn-sm btn-outline-primary" onclick="copyLogText(${index})">
                        <i data-lucide="clipboard" class="me-1"></i>复制
                    </button>
                </div>
            `;
            
            anomalyWindowsList.appendChild(listItem);
        });
        
        // 初始化图标
        lucide.createIcons();
    }
}

// 复制日志文本
function copyLogText(index) {
    if (!window.allWindowsData || !window.allWindowsData[index]) return;
    
    const logText = window.allWindowsData[index].logs.join('\n');
    
    // 使用Clipboard API
    if (navigator.clipboard) {
        navigator.clipboard.writeText(logText)
            .then(() => {
                showAlert('日志内容已复制到剪贴板', 'success');
            })
            .catch(err => {
                console.error('复制失败:', err);
                showAlert('复制失败，请手动复制', 'warning');
            });
    } else {
        // 回退方法
        const textarea = document.createElement('textarea');
        textarea.value = logText;
        textarea.style.position = 'fixed';
        textarea.style.opacity = 0;
        document.body.appendChild(textarea);
        textarea.select();
        
        try {
            const successful = document.execCommand('copy');
            if (successful) {
                showAlert('日志内容已复制到剪贴板', 'success');
            } else {
                showAlert('复制失败，请手动复制', 'warning');
            }
        } catch (err) {
            console.error('复制失败:', err);
            showAlert('复制失败，请手动复制', 'warning');
        }
        
        document.body.removeChild(textarea);
    }
}

// 切换日志文本展开/折叠
function toggleLogText(index) {
    const logText = document.getElementById(`log-text-${index}`);
    if (!logText) return;
    
    logText.classList.toggle('collapsed');
    
    // 更新图标
    const toggleIcon = event.currentTarget.querySelector('.toggle-icon');
    if (logText.classList.contains('collapsed')) {
        toggleIcon.setAttribute('name', 'chevrons-down');
    } else {
        toggleIcon.setAttribute('name', 'chevrons-up');
    }
    
    // 重新初始化图标
    lucide.createIcons({
        icons: {
            'chevrons-down': true,
            'chevrons-up': true
        }
    });
} 