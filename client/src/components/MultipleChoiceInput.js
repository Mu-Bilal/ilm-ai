import React from 'react';
import { CheckCircle } from 'lucide-react';
import { Button } from './HelperComponents';

const MultipleChoiceInput = ({ question, userAnswer, setUserAnswer, onSubmitAnswer }) => {
  // Handle radio button selection
  const handleOptionSelect = (option) => {
    setUserAnswer(option);
  };

  return (
    <>
      <div className="space-y-3 mb-4">
        {question.options.map((option, index) => (
          <div 
            key={index} 
            className={`p-3 rounded-lg border cursor-pointer transition-colors
              ${userAnswer === option 
                ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/30 dark:border-blue-400' 
                : 'border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700'}`}
            onClick={() => handleOptionSelect(option)}
          >
            <div className="flex items-center gap-3">
              <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center
                ${userAnswer === option 
                  ? 'border-blue-500 dark:border-blue-400' 
                  : 'border-gray-400 dark:border-gray-500'}`}
              >
                {userAnswer === option && (
                  <div className="w-2.5 h-2.5 rounded-full bg-blue-500 dark:bg-blue-400"></div>
                )}
              </div>
              <span className="flex-1">{option}</span>
            </div>
          </div>
        ))}
      </div>
      <Button 
        onClick={onSubmitAnswer} 
        icon={CheckCircle} 
        className="w-full" 
        disabled={!userAnswer}
      >
        Check Answer
      </Button>
    </>
  );
};

export default MultipleChoiceInput; 