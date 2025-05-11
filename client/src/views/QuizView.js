import React, { useState } from 'react';
import { CheckCircle, XCircle, ArrowLeft, Brain, Sparkles, Target, Loader2 } from 'lucide-react';
import { Card, Button } from '../components/HelperComponents';
import ChatInterface from '../components/ChatInterface';
import TextQuestionInput from '../components/TextQuestionInput';
import MultipleChoiceInput from '../components/MultipleChoiceInput';
import MultipleChoiceExplanation from '../components/MultipleChoiceExplanation';

const LoadingOverlay = () => (
  <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
    <div className="bg-white dark:bg-gray-800 p-8 rounded-xl shadow-2xl flex flex-col items-center max-w-sm w-full mx-4">
      <div className="relative">
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
        </div>
        <div className="w-16 h-16 flex items-center justify-center">
          <Brain className="w-8 h-8 text-blue-500 animate-pulse" />
        </div>
      </div>
      <h3 className="text-xl font-semibold mt-6 mb-2 text-gray-900 dark:text-gray-100">Evaluating Your Answer</h3>
      <p className="text-gray-600 dark:text-gray-400 text-center">
        Our AI is analyzing your response and providing detailed feedback...
      </p>
    </div>
  </div>
);

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
  isAnswerCorrect,
  answerEvaluation,
  isEvaluatingAnswer
}) => {
  const [localLoading, setLocalLoading] = useState(false);

  const getQuizTitle = () => {
    let title = '';
    if (quizType === 'personalized') title = `Personalized Quiz: ${courseName}`;
    else if (quizType === 'flash') title = `Flash Quiz: ${topicName || courseName}`;
    else if (quizType === 'test') title = `Test Quiz: ${topicName || courseName}`;
    return title;
  };

  const getQuizIcon = () => {
    switch (quizType) {
      case 'personalized':
        return <Brain className="w-6 h-6 text-green-500" />;
      case 'flash':
        return <Target className="w-6 h-6 text-yellow-500" />;
      case 'test':
        return <Sparkles className="w-6 h-6 text-purple-500" />;
      default:
        return <Brain className="w-6 h-6 text-blue-500" />;
    }
  };

  // Determine if question is multiple choice by checking for options property
  const isMultipleChoice = question.options && Array.isArray(question.options);
  
  const handleSubmit = async () => {
    setLocalLoading(true);
    await onSubmitAnswer();
    setLocalLoading(false);
  };

  return (
    <>
      {(isEvaluatingAnswer || localLoading) && <LoadingOverlay />}
      <Card className="max-w-3xl mx-auto">
        <div className="flex items-center gap-3 mb-4">
          {getQuizIcon()}
          <h2 className="text-2xl font-semibold">{getQuizTitle()}</h2>
        </div>
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
                onSubmitAnswer={handleSubmit}
              />
            ) : (
              <>
                <textarea
                  value={userAnswer}
                  onChange={(e) => setUserAnswer(e.target.value)}
                  placeholder="Type your answer here..."
                  rows="5"
                  className="w-full p-3 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 focus:ring-2 focus:ring-blue-500 outline-none mb-4"
                />
                <Button 
                  onClick={handleSubmit} 
                  icon={CheckCircle} 
                  className="w-full" 
                  disabled={!userAnswer.trim() || isEvaluatingAnswer || localLoading}
                >
                  {(isEvaluatingAnswer || localLoading) ? 'Evaluating...' : 'Check Answer'}
                </Button>
              </>
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
            ) : answerEvaluation ? (
              <>
                <div>
                  <h4 className="font-semibold text-gray-700 dark:text-gray-300">Your Answer:</h4>
                  <p className="p-3 bg-gray-100 dark:bg-gray-700 rounded-md">{userAnswer || "(No answer provided)"}</p>
                </div>

                <div className={`p-4 ${isAnswerCorrect ? 'bg-green-50 dark:bg-green-900/30 border-green-200 dark:border-green-700' : 'bg-red-50 dark:bg-red-900/30 border-red-200 dark:border-red-700'} border rounded-lg`}>
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      {isAnswerCorrect ? (
                        <CheckCircle className="w-5 h-5 text-green-500" />
                      ) : (
                        <XCircle className="w-5 h-5 text-red-500" />
                      )}
                      <h4 className={`font-semibold ${isAnswerCorrect ? 'text-green-700 dark:text-green-300' : 'text-red-700 dark:text-red-300'}`}>
                        {answerEvaluation.assessment}
                      </h4>
                    </div>
                    <span className={`text-sm font-medium ${isAnswerCorrect ? 'text-green-600' : 'text-red-600'}`}>
                      Score: {answerEvaluation.score}%
                    </span>
                  </div>

                  {answerEvaluation.feedback && (
                    <div className="mb-3">
                      <p className={`${isAnswerCorrect ? 'text-green-600 dark:text-green-300' : 'text-red-600 dark:text-red-300'}`}>
                        {answerEvaluation.feedback}
                      </p>
                    </div>
                  )}

                  {answerEvaluation.justification && (
                    <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
                      <h5 className="font-medium text-gray-700 dark:text-gray-300 mb-2">Detailed Explanation:</h5>
                      <p className="text-gray-600 dark:text-gray-400">{answerEvaluation.justification}</p>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div>
                <h4 className="font-semibold text-gray-700 dark:text-gray-300">Your Answer:</h4>
                <p className="p-3 bg-gray-100 dark:bg-gray-700 rounded-md">{userAnswer || "(No answer provided)"}</p>
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
                </div>
              </div>
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
    </>
  );
};

export default QuizView; 