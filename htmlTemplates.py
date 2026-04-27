css = '''
<style>
/* Modern Chat Container Styles */
.chat-message {
    padding: 1.25rem; 
    border-radius: 0.75rem; 
    margin-bottom: 1.25rem; 
    display: flex;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    animation: fadeIn 0.3s ease-in-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.chat-message.user {
    background-color: #343a40;
    border-left: 4px solid #007bff;
}

.chat-message.bot {
    background-color: #212529;
    border-left: 4px solid #6c757d;
}

.chat-message .avatar {
    width: 15%;
    display: flex;
    align-items: center;
    justify-content: center;
}

.chat-message .avatar img {
    max-width: 55px;
    max-height: 55px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid rgba(255, 255, 255, 0.1);
}

.chat-message .message {
    width: 85%;
    padding: 0 1.25rem;
    color: #e9ecef;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    line-height: 1.6;
    align-self: center;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/Fb16rkQW/chatbot-icon.avif">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/qLTM5MJz/user-chat.jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''