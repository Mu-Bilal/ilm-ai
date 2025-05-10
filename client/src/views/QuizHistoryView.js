import React from 'react';
import { Clock, Target, Brain, CheckCircle, XCircle } from 'lucide-react';
import { Card } from '../components/HelperComponents';

const QuizHistoryView = ({ course, quizHistory, onNavigate }) => {
  const formatDate = (date) => {
    return new Date(date).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getQuizTypeIcon = (type) => {
    switch (type) {
      case 'test':
        return <Brain className="w-5 h-5 text-purple-500" />;
      case 'flash':
        return <Target className="w-5 h-5 text-yellow-500" />;
      default:
        return <Target className="w-5 h-5 text-green-500" />;
    }
  };

  const getQuizTypeLabel = (type) => {
    switch (type) {
      case 'test':
        return 'Test Quiz';
      case 'flash':
        return 'Flash Quiz';
      default:
        return 'Personalized Quiz';
    }
  };

  return (
    <div className="space-y-8">
      <Card>
        <div className="flex items-center gap-3 mb-6">
          <Clock className="w-8 h-8 text-blue-500" />
          <h2 className="text-2xl font-bold">Quiz History</h2>
        </div>

        {quizHistory.length === 0 ? (
          <div className="text-center py-8">
            <p className="text-gray-600 dark:text-gray-400">No quiz attempts yet. Start learning by taking a quiz!</p>
          </div>
        ) : (
          <div className="space-y-4">
            {quizHistory.map((attempt, index) => (
              <Card key={index} className="hover:shadow-md transition-shadow">
                <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                  <div className="flex items-center gap-3">
                    {getQuizTypeIcon(attempt.type)}
                    <div>
                      <h3 className="font-semibold">{getQuizTypeLabel(attempt.type)}</h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        {attempt.topicName ? `Topic: ${attempt.topicName}` : 'Course-wide quiz'}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="text-right">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium">{attempt.score}%</span>
                        {attempt.score >= 70 ? (
                          <CheckCircle className="w-4 h-4 text-green-500" />
                        ) : (
                          <XCircle className="w-4 h-4 text-red-500" />
                        )}
                      </div>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        {attempt.correctAnswers} / {attempt.totalQuestions} correct
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {formatDate(attempt.date)}
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        Duration: {attempt.duration} min
                      </p>
                    </div>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
};

export default QuizHistoryView; 