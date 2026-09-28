"""
Custom CSS for the research tool.
"""

CUSTOM_CSS = """
/* ========================================
   General
   ======================================== */

footer { display: none !important; }

.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    border: none !important;
    box-shadow: none !important;
}

/* ========================================
   Header
   ======================================== */

#app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 4px 12px;
    border-bottom: 1px solid var(--border-color-primary);
    background: var(--background-fill-primary);
    min-height: 40px;
}

#sidebar-toggle, #result-toggle, #dark-mode-toggle {
    min-width: 36px !important;
    max-width: 36px !important;
    padding: 6px !important;
    font-size: 1.1rem;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

#sidebar-toggle:hover, #result-toggle:hover, #dark-mode-toggle:hover {
    background: var(--background-fill-secondary) !important;
}

#header-title {
    flex: 1;
    text-align: center;
    margin: 0;
}

#header-title p { margin: 0; }

/* ========================================
   Sidebar (left)
   ======================================== */

#sidebar-column {
    border-right: 1px solid var(--border-color-primary);
    padding: 8px 12px !important;
    overflow-y: auto;
    max-height: calc(100vh - 60px);
    min-width: 220px !important;
    max-width: 280px !important;
}

.section-label p { margin: 4px 0 !important; font-size: 0.8rem; }

.doc-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 4px 8px;
    border-radius: 6px;
    font-size: 0.8rem;
}

.doc-item:hover { background: var(--background-fill-secondary); }

.doc-name {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* token bar */
.token-bar-container { margin: 4px 0; }
.token-text { font-size: 0.7rem; color: var(--body-text-color-subdued); margin-bottom: 2px; }
.token-bar { height: 4px; background: var(--background-fill-secondary); border-radius: 2px; }
.token-bar-fill { height: 100%; border-radius: 2px; transition: width 0.3s; }
.token-bar-fill.normal { background: var(--color-accent); }
.token-bar-fill.warning { background: #f59e0b; }
.token-bar-fill.critical { background: #ef4444; }

/* history items */
.history-item {
    padding: 6px 8px;
    border-radius: 6px;
    font-size: 0.78rem;
    cursor: pointer;
    margin: 2px 0;
}
.history-item:hover { background: var(--background-fill-secondary); }

/* ========================================
   Chat area (middle)
   ======================================== */

#chat-column {
    padding: 0 !important;
}

#chatbot {
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}

#input-row {
    padding: 8px 12px !important;
    gap: 4px !important;
    border-top: 1px solid var(--border-color-primary);
}

#message-input {
    border-radius: 12px !important;
}
#message-input textarea {
    min-height: 44px !important;
    max-height: 400px !important;
    overflow-y: auto !important;
    resize: vertical !important;
}

#send-btn, #stop-btn {
    min-width: 50px !important;
    border-radius: 12px !important;
}

#research-btn {
    min-width: 50px !important;
    border-radius: 12px !important;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
    border: none !important;
}

#research-btn:hover {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
}

/* ========================================
   Result panel (right)
   ======================================== */

#result-panel {
    border-left: 1px solid var(--border-color-primary);
    padding: 8px 12px !important;
}

/* all 4 tab contents: scroll bar for long content */
#report-display,
#sources-display,
#progress-display,
#extracts-display {
    max-height: calc(100vh - 160px);
    overflow-y: auto;
}

#report-display {
    padding: 16px;
    font-size: 0.9rem;
    line-height: 1.6;
}

#report-display h1 { font-size: 1.4rem; margin: 16px 0 8px; }
#report-display h2 { font-size: 1.2rem; margin: 14px 0 6px; }
#report-display h3 { font-size: 1.05rem; margin: 12px 0 4px; }
#report-display code { background: var(--background-fill-secondary); padding: 1px 4px; border-radius: 3px; }
#report-display pre { background: var(--background-fill-secondary); padding: 12px; border-radius: 8px; overflow-x: auto; }
#report-display blockquote { border-left: 3px solid var(--color-accent); padding-left: 12px; margin: 8px 0; }

/* progress display */
.progress-phase {
    padding: 4px 8px;
    font-size: 0.8rem;
    border-radius: 4px;
    margin: 2px 0;
}

.progress-phase.active {
    background: var(--color-accent-soft);
    font-weight: 600;
}

.progress-phase.done {
    color: var(--body-text-color-subdued);
}

.source-item {
    padding: 3px 8px;
    font-size: 0.75rem;
    border-bottom: 1px solid var(--border-color-primary);
}

.source-item a {
    color: var(--color-accent);
    text-decoration: none;
}

.source-item a:hover { text-decoration: underline; }

/* ========================================
   Template chips
   ======================================== */

#template-chips {
    padding: 4px 12px !important;
    gap: 4px !important;
    flex-wrap: wrap;
}

#template-chips button {
    font-size: 0.75rem !important;
    padding: 4px 10px !important;
    border-radius: 16px !important;
    white-space: nowrap;
}

/* ========================================
   Hidden elements
   ======================================== */

#drop-upload, #paste-buffer {
    position: fixed !important;
    left: -9999px !important;
    top: -9999px !important;
    width: 1px !important;
    height: 1px !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

/* ========================================
   Responsive
   ======================================== */

@media (max-width: 768px) {
    #sidebar-column {
        position: absolute;
        z-index: 100;
        background: var(--background-fill-primary);
        left: 0;
        top: 44px;
        bottom: 0;
        width: 260px !important;
        max-width: 260px !important;
        box-shadow: 4px 0 20px rgba(0,0,0,0.15);
    }

    #result-panel {
        position: absolute;
        z-index: 100;
        background: var(--background-fill-primary);
        right: 0;
        top: 44px;
        bottom: 0;
        width: 90vw !important;
        max-width: 500px !important;
        box-shadow: -4px 0 20px rgba(0,0,0,0.15);
    }
}

/* ========================================
   Footer
   ======================================== */

#app-footer {
    text-align: center;
    padding: 4px 12px !important;
    border-top: 1px solid var(--border-color-primary);
}

#app-footer p { margin: 0; font-size: 0.7rem; color: var(--body-text-color-subdued); }
#app-footer a { color: var(--body-text-color-subdued); text-decoration: none; }
#app-footer a:hover { text-decoration: underline; }

/* export area (left, below the template row) */
#export-row {
    gap: 6px !important;
}
#export-file {
    min-height: 0 !important;
}

/* invisible buttons that must stay in the DOM (for JS triggers) */
.hidden-btn {
    position: absolute !important;
    width: 1px !important;
    height: 1px !important;
    overflow: hidden !important;
    opacity: 0 !important;
}
"""
