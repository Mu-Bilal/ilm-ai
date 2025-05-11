import React from 'react';
import { CheckCircle, XCircle, Check } from 'lucide-react';

const MultipleChoiceExplanation = ({ question, userAnswer, isAnswerCorrect, topicName }) => {
  return (
    <div>
      <h4 className="font-semibold text-gray-700 dark:text-gray-300 mb-3">Your Answer:</h4>
      
      <div className="space-y-3 mb-4">
        {question.options.map((option, index) => {
          const isCorrectOption = index === question.correctOptionIndex;
          const isSelectedOption = userAnswer === option;
          
          // Determine the styling based on correctness and selection
          let optionStyle = '';
          
          if (isCorrectOption) {
            // Correct answer styling
            optionStyle = 'border-green-500 bg-green-50 dark:bg-green-900/30 dark:border-green-400';
          } else if (isSelectedOption && !isCorrectOption) {
            // Wrong selection styling
            optionStyle = 'border-red-500 bg-red-50 dark:bg-red-900/30 dark:border-red-400';
          } else {
            // Unselected option styling
            optionStyle = 'border-gray-300 dark:border-gray-600 opacity-70';
          }
          
          return (
            <div 
              key={index} 
              className={`p-3 rounded-lg border transition-colors ${optionStyle}`}
            >
              <div className="flex items-center gap-3">
                <div className={`w-5 h-5 flex items-center justify-center`}>
                  {isCorrectOption && (
                    <CheckCircle className="w-5 h-5 text-green-500" />
                  )}
                  {isSelectedOption && !isCorrectOption && (
                    <XCircle className="w-5 h-5 text-red-500" />
                  )}
                </div>
                <span className="flex-1">{option}</span>
              </div>
            </div>
          );
        })}
      </div>
      
      <div className={`p-4 mt-4 ${isAnswerCorrect ? 'bg-green-50 dark:bg-green-900/30 border-green-200 dark:border-green-700' : 'bg-red-50 dark:bg-red-900/30 border-red-200 dark:border-red-700'} border rounded-lg`}>
        <div className="flex items-center gap-2 mb-2">
          {isAnswerCorrect ? (
            <CheckCircle className="w-5 h-5 text-green-500" />
          ) : (
            <XCircle className="w-5 h-5 text-red-500" />
          )}
          <h4 className={`font-semibold ${isAnswerCorrect ? 'text-green-700 dark:text-green-300' : 'text-red-700 dark:text-red-300'}`}>
            {isAnswerCorrect ? 'Correct!' : 'Not quite right'}
          </h4>
        </div>
        <p className={`${isAnswerCorrect ? 'text-green-600 dark:text-green-300' : 'text-red-600 dark:text-red-300'}`}>
          {question.explanation}
        </p>
        {question.type === 'test' && topicName && (
          <p className="text-xs text-blue-500 dark:text-blue-400 mt-2">
            (Related to: <a href="#" className="underline">{topicName} course materials</a>)
          </p>
        )}
      </div>
    </div>
  );
};

export default MultipleChoiceExplanation; 