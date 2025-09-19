<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>The Adaptive Loyalist AI</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/feather-icons"></script>
    <script src="https://cdn.jsdelivr.net/npm/feather-icons/dist/feather.min.js"></script>
    <link href="https://unpkg.com/aos@2.3.1/dist/aos.css" rel="stylesheet">
    <script src="https://unpkg.com/aos@2.3.1/dist/aos.js"></script>
    <style>
        .bg-burgundy-50 { background-color: #1A0A0A; }
        .bg-burgundy-100 { background-color: #2A1010; }
        .bg-burgundy-200 { background-color: #3A1515; }
        .bg-burgundy-300 { background-color: #5A1F1F; }
        .bg-burgundy-400 { background-color: #7A2A2A; }
        .bg-burgundy-500 { background-color: #9A3535; }
        .bg-burgundy-600 { background-color: #BA4040; }
        .bg-burgundy-700 { background-color: #DA4B4B; }
        .text-burgundy-500 { color: #BA4040; }
        .border-burgundy-300 { border-color: #5A1F1F; }
        .hover\:bg-burgundy-100:hover { background-color: #3A1515; }
    </style>
</head>
<body class="bg-gray-900 min-h-screen bg-opacity-90 text-gray-100">
    <div class="max-w-4xl mx-auto p-6">
        <!-- Header -->
        <header class="mb-8 text-center" data-aos="fade-down">
            <div class="flex justify-between items-start mb-4">
                <div class="relative">
                    <button id="settings-btn" class="p-2 rounded-full hover:bg-burgundy-100">
                        <i data-feather="settings" class="w-6 h-6 text-burgundy-500"></i>
                    </button>
                    <div id="settings-menu" class="hidden absolute left-0 mt-2 w-48 bg-white rounded-md shadow-lg py-1 z-10 border border-burgundy-200">
                        <a href="#" class="block px-4 py-2 text-sm text-gray-700 hover:bg-burgundy-50">Login</a>
                        <a href="#" class="block px-4 py-2 text-sm text-gray-700 hover:bg-burgundy-50">Dark Mode</a>
                        <a href="#" class="block px-4 py-2 text-sm text-gray-700 hover:bg-burgundy-50">Language</a>
                        <a href="#" class="block px-4 py-2 text-sm text-gray-700 hover:bg-burgundy-50">Get Help</a>
                    </div>
                </div>
                <div class="flex flex-col items-center flex-grow">
                    <div class="flex items-center justify-center mb-4">
                        <div class="relative w-12 h-12 mr-3">
                            <div class="absolute inset-0 rounded-full bg-burgundy-400 animate-pulse"></div>
                            <div class="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                                <svg class="w-8 h-8 text-white animate-flap" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M22 8s-1.5-2-4.5-2S13 8 13 8s-1.5-2-4.5-2S4 8 4 8s1.5 2 4.5 2c1.5 0 2.5-.5 3.5-1v6c-1 .5-2 1-3.5 1-3 0-4.5 2-4.5 2s1.5 2 4.5 2 4.5-2 4.5-2v-6c1 .5 2 1 3.5 1 3 0 4.5-2 4.5-2z" fill="#FFF"/>
                                    <path d="M12 14v-6" stroke="#FFF" stroke-width="2" stroke-linecap="round"/>
                                </svg>
                            </div>
                        </div>
                        <h1 class="text-4xl font-bold text-burgundy-600">The Adaptive Loyalist AI</h1>
                    </div>
                </div>
                <div class="w-10"></div> <!-- Spacer to balance the layout -->
            </div>
            <p class="text-lg text-burgundy-400 italic">"A sensible, matured and highly cognitive companion"</p>
            <div class="mt-4 p-3 bg-gray-800 bg-opacity-70 rounded-lg inline-block backdrop-blur-sm">
                <span class="text-burgundy-400 font-medium">
                    <i data-feather="database" class="inline mr-2 w-5 h-5"></i>
                    Memory Status: <span class="text-green-700">Online</span> | Total Memories: <span id="memory-count">0</span>
                </span>
            </div>
        </header>

        <!-- Chat Header -->
        <div class="flex items-center justify-between border-b border-burgundy-200 mb-6">
            <button class="tab-btn active px-6 py-3 text-burgundy-600 font-medium border-b-2 border-burgundy-500 flex items-center">
                Chat
            </button>
            <button id="new-chat-btn" class="p-2 rounded-full hover:bg-burgundy-100 text-burgundy-500">
                <i data-feather="plus" class="w-5 h-5"></i>
            </button>
        </div>

        <!-- Chat Container -->
        <div id="chat-tab" class="tab-content bg-gray-800 bg-opacity-90 rounded-xl shadow-lg overflow-hidden border border-burgundy-300 backdrop-blur-sm" data-aos="fade-up">
            <!-- Chat Messages -->
            <div id="chat-container" class="h-96 p-4 overflow-y-auto space-y-4">
                <!-- Messages will be inserted here by JavaScript -->
                <div class="text-center py-10 text-burgundy-300">
                    <i data-feather="message-square" class="w-12 h-12 mx-auto mb-3"></i>
                    <p>Start a conversation with your AI companion</p>
                </div>
            </div>

            <!-- Input Area -->
            <div class="border-t border-burgundy-200 p-4 bg-burgundy-50 bg-opacity-70 backdrop-blur-sm">
                <div class="flex">
                    <input 
                        id="user-input" 
                        type="text" 
                        placeholder="What's buggin' ya?" 
                        class="flex-1 px-4 py-3 rounded-l-lg border border-burgundy-300 focus:outline-none focus:ring-2 focus:ring-burgundy-400 bg-gray-700 text-white placeholder-gray-400"
                    >
                    <button 
                        id="send-button" 
                        class="bg-burgundy-500 text-white px-6 py-3 rounded-r-lg hover:bg-burgundy-600 transition-colors flex items-center"
                    >
                        <i data-feather="send" class="mr-2 w-5 h-5"></i> Send
                    </button>
                </div>
            </div>
        </div>

        <!-- Features Section -->
        <div class="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6" data-aos="fade-up">
            <div class="bg-gray-800 bg-opacity-90 p-6 rounded-lg shadow border border-burgundy-200 hover:shadow-lg transition-all backdrop-blur-sm">
                <div class="text-burgundy-500 mb-4">
                    <i data-feather="globe" class="w-8 h-8"></i>
                </div>
                <h3 class="text-xl font-semibold text-burgundy-600 mb-2">Bilingual AI</h3>
                <p class="text-gray-300">Fluent in both English and Hindi, responding naturally in the language you prefer.</p>
            </div>
            <div class="bg-gray-800 bg-opacity-90 p-6 rounded-lg shadow border border-burgundy-200 hover:shadow-lg transition-all backdrop-blur-sm">
                <div class="text-burgundy-500 mb-4">
                    <i data-feather="book" class="w-8 h-8"></i>
                </div>
                <h3 class="text-xl font-semibold text-burgundy-600 mb-2">Context Aware</h3>
                <p class="text-gray-300">Remembers both long-term knowledge and short-term conversation history.</p>
            </div>
            <div class="bg-white bg-opacity-90 p-6 rounded-lg shadow border border-burgundy-200 hover:shadow-lg transition-all backdrop-blur-sm">
                <div class="text-burgundy-500 mb-4">
                    <i data-feather="shield" class="w-8 h-8"></i>
                </div>
                <h3 class="text-xl font-semibold text-burgundy-600 mb-2">Privacy Focused</h3>
                <p class="text-gray-300">Conversations are securely logged for improvement while respecting your privacy.</p>
            </div>
        </div>
    </div>


    <script>
        // Initialize AOS and Feather Icons
        // Settings menu toggle
        document.getElementById('settings-btn').addEventListener('click', function() {
            const menu = document.getElementById('settings-menu');
            menu.classList.toggle('hidden');
        });

        // Close menu when clicking outside
        document.addEventListener('click', function(event) {
            const settingsBtn = document.getElementById('settings-btn');
            const settingsMenu = document.getElementById('settings-menu');
            if (!settingsBtn.contains(event.target) && !settingsMenu.contains(event.target)) {
                settingsMenu.classList.add('hidden');
            }
        });

        // New chat button functionality
        document.getElementById('new-chat-btn').addEventListener('click', function() {
            const chatContainer = document.getElementById('chat-container');
            chatContainer.innerHTML = `
                <div class="text-center py-10 text-burgundy-300">
                    <i data-feather="message-square" class="w-12 h-12 mx-auto mb-3"></i>
                    <p>Start a new conversation with your AI companion</p>
                </div>
            `;
            feather.replace();
        });
        AOS.init({
            duration: 800,
            easing: 'ease-in-out'
        });

        // Add custom flap animation
        const style = document.createElement('style');
        style.innerHTML = `
            @keyframes flap {
                0% { transform: translateY(0) rotate(0deg); }
                50% { transform: translateY(-5px) rotate(10deg); }
                100% { transform: translateY(0) rotate(0deg); }
            }
            .animate-flap {
                animation: flap 0.5s infinite alternate;
            }
        `;
        document.head.appendChild(style);
        feather.replace();
        
        // Set memory count (simulated)
        document.getElementById('memory-count').textContent = '256';
        
        // Chat functionality (simulated)
        document.getElementById('send-button').addEventListener('click', function() {
            const input = document.getElementById('user-input');
            const message = input.value.trim();
            
            if (message) {
                // Add user message
                addMessage('user', message);
                input.value = '';
                
                // Simulate AI response after a delay
                setTimeout(() => {
                    const responses = [
                        "I understand what you're saying. Let me think about that...",
                        "That's an interesting perspective. Here's what I think...",
                        "Based on my knowledge and our previous conversation, I'd say...",
                        "I appreciate you sharing that with me. My thoughts are..."
                    ];
                    const randomResponse = responses[Math.floor(Math.random() * responses.length)];
                    addMessage('assistant', randomResponse);
                }, 1000);
            }
        });
        
        function addMessage(role, content) {
            const chatContainer = document.getElementById('chat-container');
            
            // Clear initial placeholder if it exists
            if (chatContainer.querySelector('.text-center')) {
                chatContainer.innerHTML = '';
            }
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `flex ${role === 'user' ? 'justify-end' : 'justify-start'}`;
            
            const bubble = document.createElement('div');
            bubble.className = `max-w-xs md:max-w-md rounded-lg p-4 ${
                role === 'user' 
                    ? 'bg-burgundy-500 text-white rounded-br-none' 
                    : 'bg-burgundy-100 text-burgundy-700 rounded-bl-none'
            }`;
            bubble.textContent = content;
            
            messageDiv.appendChild(bubble);
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    </script>
</body>
</html>
