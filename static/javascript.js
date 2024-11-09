let sessionId = null;
const serverUrl = window.location.origin;

// Image handling
document.getElementById('imageInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('preview').src = e.target.result;
            document.getElementById('uploadBtn').disabled = false;
        };
        reader.readAsDataURL(file);
    }
});

document.getElementById('uploadBtn').addEventListener('click', async function() {
    const file = document.getElementById('imageInput').files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${serverUrl}/api/upload-and-classify/`, {
            method: 'POST',
            body: formData,
            credentials: 'same-origin'
        });

        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

        const result = await response.json();
        sessionId = result.session_id;
        
        appendMessage(`Image classified! Class: ${result.predicted_class}`);
        document.getElementById('messageInput').disabled = false;
        document.getElementById('sendBtn').disabled = false;
    } catch (error) {
        console.error('Upload error:', error);
        appendMessage(`Error uploading image: ${error.message}`);
    }
});

// Chat handling
document.getElementById('sendBtn').addEventListener('click', sendMessage);
document.getElementById('messageInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') sendMessage();
});

async function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    const message = messageInput.value.trim();
    
    if (!message || !sessionId) return;

    appendMessage(`You: ${message}`);
    messageInput.value = '';

    try {
        const response = await fetch(`${serverUrl}/api/chat/${sessionId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message }),
            credentials: 'same-origin'
        });

        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

        const result = await response.json();
        appendMessage(`Server: ${result.llm_response}`);
    } catch (error) {
        console.error('Chat error:', error);
        appendMessage(`Error sending message: ${error.message}`);
    }
}

function appendMessage(message) {
    const chatDisplay = document.getElementById('chatDisplay');
    const messageElement = document.createElement('div');
    messageElement.textContent = message;
    chatDisplay.appendChild(messageElement);
    chatDisplay.scrollTop = chatDisplay.scrollHeight;
}
