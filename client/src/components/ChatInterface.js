import React, { useState } from 'react';
import { Send } from 'lucide-react';
import { Button } from './HelperComponents';

const ChatInterface = () => {
  const [messages, setMessages] = useState([
    { id: 1, text: "Need help understanding this concept?", sender: "system" }
  ]);
  const [newMessage, setNewMessage] = useState('');

  const handleSendMessage = () => {
    if (!newMessage.trim()) return;
    
    // Add user message
    const userMessage = { id: messages.length + 1, text: newMessage, sender: "user" };
    setMessages([...messages, userMessage]);
    setNewMessage('');
    
    // Simulate AI response with hardcoded responses
    setTimeout(() => {
      const responses = [
        "That's a great question! This concept relates to the fundamental principles we covered earlier.",
        "I'd recommend reviewing the section on this topic in your learning materials.",
        "Think about how this connects to what we learned about related topics.",
        "Let me clarify this for you. The key thing to understand is the relationship between these concepts.",
        "You're on the right track! Consider approaching this from a different angle."
      ];
      
      const randomResponse = responses[Math.floor(Math.random() * responses.length)];
      const botMessage = { id: messages.length + 2, text: randomResponse, sender: "system" };
      setMessages(prevMessages => [...prevMessages, botMessage]);
    }, 1000);
  };

  return (
    <div className="border rounded-lg overflow-hidden">
      <div className="p-3 bg-blue-50 dark:bg-blue-900/30 border-b">
        <h3 className="font-medium text-blue-800 dark:text-blue-300">Discussion Assistant</h3>
      </div>
      
      <div className="h-64 overflow-y-auto p-3 space-y-3 bg-gray-50 dark:bg-gray-800">
        {messages.map(message => (
          <div 
            key={message.id} 
            className={`p-2 rounded-lg max-w-[80%] ${
              message.sender === 'user' 
                ? 'ml-auto bg-blue-100 dark:bg-blue-800 text-blue-900 dark:text-blue-100' 
                : 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200'
            }`}
          >
            {message.text}
          </div>
        ))}
      </div>
      
      <div className="p-3 border-t flex gap-2">
        <input
          type="text"
          value={newMessage}
          onChange={(e) => setNewMessage(e.target.value)}
          placeholder="Ask a question about this topic..."
          className="flex-1 p-2 rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 focus:ring-2 focus:ring-blue-500 outline-none"
          onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
        />
        <Button 
          onClick={handleSendMessage} 
          icon={Send} 
          disabled={!newMessage.trim()}
          className="px-3"
        >
          Send
        </Button>
      </div>
    </div>
  );
};

export default ChatInterface; 