import React from 'react';
import { CheckCircle, XCircle, ArrowLeft } from 'lucide-react';
import { Card, Button } from '../components/HelperComponents';
import ChatInterface from '../components/ChatInterface';
import TextQuestionInput from '../components/TextQuestionInput';
import MultipleChoiceInput from '../components/MultipleChoiceInput';
import MultipleChoiceExplanation from '../components/MultipleChoiceExplanation';

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
  totalQuestions,
  isAnswerCorrect
}) => {
  const getQuizTitle = () => {
    let title = '';
    if (quizType === 'personalized') title = `Personalized Quiz: ${courseName}`;
    else if (quizType === 'flash') title = `Flash Quiz: ${topicName || courseName}`;
    else if (quizType === 'test') title = `Test Quiz: ${topicName || courseName}`;
    return title;
  };

  // Determine if question is multiple choice by checking for options property
  const isMultipleChoice = question.options && Array.isArray(question.options);
  
  return (
    <Card className="max-w-3xl mx-auto">
      <h2 className="text-2xl font-semibold mb-1 text-center">{getQuizTitle()}</h2>
      <p className="text-sm text-gray-500 dark:text-gray-400 text-center mb-6">Question {currentQuestionIndex + 1} of {totalQuestions}</p>
      
      <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg min-h-[100px]">
        <p className="text-lg font-medium">{question.text}</p>
      </div>

      {!showExplanation && (
        <>
          {isMultipleChoice ? (
            <MultipleChoiceInput 
              question={question}
              userAnswer={userAnswer}
              setUserAnswer={setUserAnswer}
              onSubmitAnswer={onSubmitAnswer}
            />
          ) : (
            <TextQuestionInput
              userAnswer={userAnswer}
              setUserAnswer={setUserAnswer}
              onSubmitAnswer={onSubmitAnswer}
            />
          )}
        </>
      )}

      {showExplanation && (
        <div className="space-y-4">
          {isMultipleChoice ? (
            <MultipleChoiceExplanation 
              question={question}
              userAnswer={userAnswer}
              isAnswerCorrect={isAnswerCorrect}
              topicName={topicName}
            />
          ) : (
            <>
              <div>
                <h4 className="font-semibold text-gray-700 dark:text-gray-300">Your Answer:</h4>
                <p className="p-3 bg-gray-100 dark:bg-gray-700 rounded-md">{userAnswer || "(No answer provided)"}</p>
              </div>
              <div className={`p-4 ${isAnswerCorrect ? 'bg-green-50 dark:bg-green-900/30 border-green-200 dark:border-green-700' : 'bg-red-50 dark:bg-red-900/30 border-red-200 dark:border-red-700'} border rounded-lg`}>
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
            </>
          )}

          <div className="mt-6">
            <h4 className="font-semibold text-gray-700 dark:text-gray-300 mb-3">
              Still have questions? Chat with our assistant:
            </h4>
            <ChatInterface />
          </div>

          <Button 
            onClick={onNextQuestion} 
            icon={isLastQuestion ? CheckCircle : ArrowLeft} 
            className={`w-full ${isAnswerCorrect ? 'bg-green-500 hover:bg-green-600' : 'bg-blue-500 hover:bg-blue-600'}`}
          >
            {isLastQuestion ? 'Finish Quiz' : 'Next Question'}
          </Button>
        </div>
      )}
    </Card>
  );
};

export default QuizView; 