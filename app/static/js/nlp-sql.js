/**
 * NLP到SQL转换前端交互逻辑
 * 处理自然语言查询的输入、发送、展示以及历史记录管理
 */

// 全局变量
let queryHistory = JSON.parse(localStorage.getItem('nlpQueryHistory') || '[]');
const MAX_HISTORY = 10;

// 页面初始化
document.addEventListener('DOMContentLoaded', () => {
    // 初始化图标
    initIcons();
    
    // 初始化示例查询点击事件
    initExampleQueries();
    
    // 加载历史记录
    loadHistory();
    
    // 绑定事件处理函数
    document.getElementById('submitQueryBtn').addEventListener('click', handleQuerySubmit);
    document.getElementById('clearQueryBtn').addEventListener('click', clearQuery);
    document.getElementById('copyBtn').addEventListener('click', copySqlToClipboard);
    document.getElementById('clearHistoryBtn').addEventListener('click', clearHistory);
    
    // 键盘快捷键
    document.getElementById('queryInput').addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            handleQuerySubmit();
        }
    });
});

/**
 * 初始化Lucide图标
 */
function initIcons() {
    const icons = document.querySelectorAll('.icon');
    icons.forEach(icon => {
        const iconName = icon.getAttribute('data-icon');
        if (iconName) {
            try {
                lucide.createIcons({
                    icons: {
                        [iconName]: lucide[iconName] 
                    },
                    attrs: {
                        class: icon.getAttribute('class')
                    }
                });
            } catch (err) {
                console.warn(`未能加载图标: ${iconName}`, err);
            }
        }
    });
}

/**
 * 初始化示例查询点击事件
 */
function initExampleQueries() {
    const examples = document.querySelectorAll('.example-query');
    examples.forEach(example => {
        example.addEventListener('click', () => {
            const queryText = example.getAttribute('data-query');
            document.getElementById('queryInput').value = queryText;
            // 自动滚动到查询输入区域
            document.getElementById('queryInput').scrollIntoView({ behavior: 'smooth' });
            document.getElementById('queryInput').focus();
        });
    });
}

/**
 * 处理查询提交
 */
async function handleQuerySubmit() {
    const queryInput = document.getElementById('queryInput');
    const query = queryInput.value.trim();
    
    if (!query) {
        showAlert('请输入自然语言查询', 'warning');
        return;
    }
    
    // 显示加载状态
    toggleLoading(true);
    clearResults(); // 清空之前的结果
    
    try {
        const response = await fetch('/nlp/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query })
        });
        
        if (!response.ok) {
            throw new Error(`服务器响应错误: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            showAlert(`查询错误: ${data.error}`, 'danger');
            return;
        }
        
        // 显示结果
        displayResults(data);
        
        // 添加到历史记录
        addToHistory(data);
        
    } catch (error) {
        showAlert(`查询失败: ${error.message}`, 'danger');
        console.error('查询失败:', error);
    } finally {
        toggleLoading(false);
    }
}

/**
 * 显示查询结果
 */
function displayResults(data) {
    // 显示SQL结果区域
    const resultsPanel = document.getElementById('resultsPanel');
    resultsPanel.classList.remove('d-none');
    
    // 显示SQL查询
    const sqlDisplay = document.getElementById('sqlDisplay');
    sqlDisplay.textContent = data.sql;
    
    // 显示查询结果
    displayQueryResults(data.results);
}

/**
 * 显示查询结果数据表格
 */
function displayQueryResults(results) {
    const resultsContainer = document.getElementById('queryResults');
    resultsContainer.innerHTML = '';
    
    if (!results || results.length === 0) {
        resultsContainer.innerHTML = '<div class="alert alert-info">查询未返回结果</div>';
        return;
    }
    
    // 创建表格
    const table = document.createElement('table');
    table.className = 'table table-striped table-bordered';
    
    // 创建表头
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    const keys = Object.keys(results[0]);
    
    keys.forEach(key => {
        const th = document.createElement('th');
        th.textContent = key;
        headerRow.appendChild(th);
    });
    
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // 创建表体
    const tbody = document.createElement('tbody');
    
    results.forEach(row => {
        const tr = document.createElement('tr');
        
        keys.forEach(key => {
            const td = document.createElement('td');
            td.textContent = row[key] !== null ? row[key] : 'NULL';
            tr.appendChild(td);
        });
        
        tbody.appendChild(tr);
    });
    
    table.appendChild(tbody);
    resultsContainer.appendChild(table);
}

/**
 * 添加到历史记录
 */
function addToHistory(data) {
    const historyItem = {
        id: Date.now(),
        query: data.original_query,
        sql: data.sql,
        results: data.results,
        timestamp: new Date().toLocaleString()
    };
    
    // 添加到历史记录开头
    queryHistory.unshift(historyItem);
    
    // 限制历史记录数量
    if (queryHistory.length > MAX_HISTORY) {
        queryHistory = queryHistory.slice(0, MAX_HISTORY);
    }
    
    // 保存到localStorage
    localStorage.setItem('nlpQueryHistory', JSON.stringify(queryHistory));
    
    // 刷新历史记录显示
    loadHistory();
}

/**
 * 加载历史记录
 */
function loadHistory() {
    const historyContainer = document.getElementById('historyContainer');
    historyContainer.innerHTML = '';
    
    if (queryHistory.length === 0) {
        historyContainer.innerHTML = '<div class="alert alert-secondary">暂无查询历史</div>';
        return;
    }
    
    queryHistory.forEach(item => {
        const historyCard = document.createElement('div');
        historyCard.className = 'card mb-3';
        
        historyCard.innerHTML = `
            <div class="card-header d-flex justify-content-between">
                <div>
                    <strong>${item.query}</strong>
                </div>
                <small class="text-muted">${item.timestamp}</small>
            </div>
            <div class="card-body">
                <div class="mb-2"><strong>SQL:</strong> <code>${item.sql}</code></div>
                <div><strong>结果:</strong> ${item.results.length}条记录</div>
            </div>
            <div class="card-footer">
                <button class="btn btn-sm btn-primary rerun-query" data-query="${item.query}">
                    重新执行
                </button>
                <button class="btn btn-sm btn-outline-secondary remove-history" data-id="${item.id}">
                    删除
                </button>
            </div>
        `;
        
        historyContainer.appendChild(historyCard);
    });
    
    // 绑定历史记录操作按钮事件
    document.querySelectorAll('.rerun-query').forEach(btn => {
        btn.addEventListener('click', () => {
            const query = btn.getAttribute('data-query');
            document.getElementById('queryInput').value = query;
            document.getElementById('queryInput').scrollIntoView({ behavior: 'smooth' });
            handleQuerySubmit();
        });
    });
    
    document.querySelectorAll('.remove-history').forEach(btn => {
        btn.addEventListener('click', () => {
            const id = parseInt(btn.getAttribute('data-id'));
            removeHistoryItem(id);
        });
    });
}

/**
 * 删除历史记录项
 */
function removeHistoryItem(id) {
    queryHistory = queryHistory.filter(item => item.id !== id);
    localStorage.setItem('nlpQueryHistory', JSON.stringify(queryHistory));
    loadHistory();
}

/**
 * 清空所有历史记录
 */
function clearHistory() {
    if (confirm('确定要清空所有查询历史吗？')) {
        queryHistory = [];
        localStorage.removeItem('nlpQueryHistory');
        loadHistory();
        showAlert('查询历史已清空', 'info');
    }
}

/**
 * 复制SQL到剪贴板
 */
function copySqlToClipboard() {
    const sqlDisplay = document.getElementById('sqlDisplay');
    const sqlText = sqlDisplay.textContent;
    
    if (!sqlText) {
        showAlert('没有SQL可复制', 'warning');
        return;
    }
    
    navigator.clipboard.writeText(sqlText)
        .then(() => {
            showAlert('SQL已复制到剪贴板', 'success');
        })
        .catch(err => {
            showAlert('复制失败: ' + err, 'danger');
        });
}

/**
 * 清空查询输入
 */
function clearQuery() {
    document.getElementById('queryInput').value = '';
    clearResults();
}

/**
 * 清空结果显示
 */
function clearResults() {
    document.getElementById('resultsPanel').classList.add('d-none');
    document.getElementById('sqlDisplay').textContent = '';
    document.getElementById('queryResults').innerHTML = '';
}

/**
 * 切换加载状态
 */
function toggleLoading(isLoading) {
    const submitBtn = document.getElementById('submitQueryBtn');
    const spinner = submitBtn.querySelector('.spinner-border');
    const btnText = submitBtn.querySelector('.btn-text');
    
    if (isLoading) {
        spinner.classList.remove('d-none');
        btnText.textContent = '处理中...';
        submitBtn.disabled = true;
    } else {
        spinner.classList.add('d-none');
        btnText.textContent = '执行查询';
        submitBtn.disabled = false;
    }
}

/**
 * 显示提示信息
 */
function showAlert(message, type = 'info') {
    const alertsContainer = document.getElementById('alertsContainer');
    
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    alertsContainer.appendChild(alert);
    
    // 5秒后自动消失
    setTimeout(() => {
        alert.classList.remove('show');
        setTimeout(() => {
            alert.remove();
        }, 150);
    }, 5000);
} 