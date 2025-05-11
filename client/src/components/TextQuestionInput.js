import React from 'react';
import { CheckCircle } from 'lucide-react';
import { Button } from './HelperComponents';

const TextQuestionInput = ({ userAnswer, setUserAnswer, onSubmitAnswer }) => {
  return (
    <>
      <textarea
        value={userAnswer}
        onChange={(e) => setUserAnswer(e.target.value)}
        placeholder="Type your answer here..."
        rows="5"
        className="w-full p-3 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 focus:ring-2 focus:ring-blue-500 outline-none mb-4"
      />
      <Button onClick={onSubmitAnswer} icon={CheckCircle} className="w-full" disabled={!userAnswer.trim()}>
        Check Answer
      </Button>
    </>
  );
};

export default TextQuestionInput; 