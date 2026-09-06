css = '''
<style>
.chat-message {
    padding: 1rem 1.25rem;
    border-radius: 0.75rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: flex-start;
    gap: 0.875rem;
    animation: fadeIn 0.25s ease-in-out;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Light theme */
.chat-message.user {
    background-color: #e8f0fe;
    border-left: 4px solid #4285f4;
}
.chat-message.bot {
    background-color: #f1f3f4;
    border-left: 4px solid #5f6368;
}

/* Dark theme override */
@media (prefers-color-scheme: dark) {
    .chat-message.user { background-color: #1e3a5f; border-left-color: #4285f4; }
    .chat-message.bot  { background-color: #2d2d2d; border-left-color: #9aa0a6; }
}
[data-theme="dark"] .chat-message.user { background-color: #1e3a5f; border-left-color: #4285f4; }
[data-theme="dark"] .chat-message.bot  { background-color: #2d2d2d; border-left-color: #9aa0a6; }

.chat-message .avatar {
    flex-shrink: 0;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
    line-height: 1;
}

.chat-message.user .avatar { background-color: #4285f4; }
.chat-message.bot  .avatar { background-color: #5f6368; }

.chat-message .message {
    flex: 1;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 0.95rem;
    line-height: 1.65;
    color: inherit;
    white-space: pre-wrap;
    word-break: break-word;
}

/* Sources section — visually separate from the answer body */
.chat-message .message .sources-block {
    margin-top: 0.75rem;
    padding-top: 0.6rem;
    border-top: 1px solid rgba(128,128,128,0.25);
    font-size: 0.82rem;
    color: #5f6368;
    font-style: italic;
}
[data-theme="dark"] .chat-message .message .sources-block { color: #9aa0a6; }
@media (prefers-color-scheme: dark) {
    .chat-message .message .sources-block { color: #9aa0a6; }
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">🤖</div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">👤</div>
    <div class="message">{{MSG}}</div>
</div>
'''
