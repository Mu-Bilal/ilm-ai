import React from 'react';
import { CheckCircle, ArrowLeft } from 'lucide-react';
import { Card, Button } from '../components/HelperComponents';

const QuizView = ({ 
  question, 
  userAnswer, 
  setUserAnswer, 
  showExplanation, 
  onSubmitAnswer, 
  onNextQuestion, 
  isLastQuestion, 
  quizType, 
  courseName, 
  topicName,
  currentQuestionIndex,
  totalQuestions
}) => {
  const getQuizTitle = () => {
    let title = '';
    if (quizType === 'personalized') title = `Personalized Quiz: ${courseName}`;
    else if (quizType === 'flash') title = `Flash Quiz: ${topicName || courseName}`;
    else if (quizType === 'test') title = `Test Quiz: ${topicName || courseName}`;
    return title;
  };
  
  return (
    <Card className="max-w-3xl mx-auto">
      <h2 className="text-2xl font-semibold mb-1 text-center">{getQuizTitle()}</h2>
      <p className="text-sm text-gray-500 dark:text-gray-400 text-center mb-6">Question {currentQuestionIndex + 1} of {totalQuestions}</p>
      
      <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg min-h-[100px]">
        <p className="text-lg font-medium">{question.text}</p>
      </div>

      {!showExplanation && (
        <>
          <textarea
            value={userAnswer}
            onChange={(e) => setUserAnswer(e.target.value)}
            placeholder="Type your answer here..."
            rows="5"
            className="w-full p-3 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 focus:ring-2 focus:ring-blue-500 outline-none mb-4"
          />
          <Button onClick={onSubmitAnswer} icon={CheckCircle} className="w-full" disabled={!userAnswer.trim()}>Check Answer</Button>
        </>
      )}

      {showExplanation && (
        <div className="space-y-4">
          <div>
            <h4 className="font-semibold text-gray-700 dark:text-gray-300">Your Answer:</h4>
            <p className="p-3 bg-gray-100 dark:bg-gray-700 rounded-md">{userAnswer || "(No answer provided)"}</p>
          </div>
          <div className="p-4 bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-700 rounded-lg">
            <h4 className="font-semibold text-blue-700 dark:text-blue-300 mb-1">Explanation:</h4>
            <p className="text-blue-600 dark:text-blue-300">{question.explanation}</p>
            {question.type === 'test' && topicName && (
                <p className="text-xs text-blue-500 dark:text-blue-400 mt-2">
                    (Related to: <a href="#" className="underline">{topicName} course materials</a>)
                </p>
            )}
          </div>
          <Button onClick={onNextQuestion} icon={isLastQuestion ? CheckCircle : ArrowLeft} className="w-full bg-green-500 hover:bg-green-600">
            {isLastQuestion ? 'Finish Quiz' : 'Next Question'}
          </Button>
        </div>
      )}
    </Card>
  );
};

export default QuizView; 