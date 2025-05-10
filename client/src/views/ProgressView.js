import React from 'react';
import { BarChart3, Target, Brain } from 'lucide-react';
import { Card, ProgressBar } from '../components/HelperComponents';

const ProgressView = ({ course }) => {
  // Calculate topic dependencies and relationships
  const topicDependencies = {
    'topic1_1': [], // Introduction to Probability has no dependencies
    'topic1_2': ['topic1_1'], // Bayesian Networks depends on Introduction to Probability
    'topic1_3': ['topic1_1', 'topic1_2'], // Hidden Markov Models depends on both previous topics
  };

  // Calculate mastery levels
  const getMasteryLevel = (progress) => {
    if (progress >= 90) return { level: 'Mastered', color: 'text-green-500' };
    if (progress >= 70) return { level: 'Proficient', color: 'text-blue-500' };
    if (progress >= 40) return { level: 'Developing', color: 'text-yellow-500' };
    return { level: 'Beginner', color: 'text-red-500' };
  };

  return (
    <div className="space-y-8">
      <Card>
        <div className="flex items-center gap-3 mb-6">
          <BarChart3 className="w-8 h-8 text-blue-500" />
          <h2 className="text-2xl font-bold">Course Progress Analysis</h2>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Overall Progress */}
          <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
            <h3 className="text-lg font-semibold mb-3">Overall Progress</h3>
            <div className="flex items-center gap-4">
              <div className="w-24 h-24 rounded-full border-4 border-blue-500 flex items-center justify-center">
                <span className="text-2xl font-bold">{course.progress}%</span>
              </div>
              <div className="flex-1">
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Course Completion</p>
                <ProgressBar progress={course.progress} size="h-3" color="bg-blue-500" />
              </div>
            </div>
          </div>

          {/* Topic Mastery */}
          <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
            <h3 className="text-lg font-semibold mb-3">Topic Mastery</h3>
            <div className="space-y-4">
              {course.topics.map(topic => {
                const mastery = getMasteryLevel(topic.progress);
                return (
                  <div key={topic.id} className="flex items-center gap-3">
                    <div className="w-16 text-right">
                      <span className={`text-sm font-medium ${mastery.color}`}>{mastery.level}</span>
                    </div>
                    <div className="flex-1">
                      <div className="flex justify-between text-sm mb-1">
                        <span>{topic.name}</span>
                        <span>{topic.progress}%</span>
                      </div>
                      <ProgressBar progress={topic.progress} size="h-2" color={`bg-${mastery.color.split('-')[1]}-500`} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </Card>

      {/* Knowledge Graph */}
      <Card>
        <h3 className="text-xl font-semibold mb-4">Knowledge Dependencies</h3>
        <div className="relative h-64 bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="grid grid-cols-3 gap-8">
              {course.topics.map((topic, index) => (
                <div key={topic.id} className="relative">
                  <div className={`p-4 rounded-lg ${index === 0 ? 'bg-green-100 dark:bg-green-900' : index === 1 ? 'bg-blue-100 dark:bg-blue-900' : 'bg-purple-100 dark:bg-purple-900'}`}>
                    <div className="flex items-center gap-2 mb-2">
                      <Brain className="w-5 h-5" />
                      <span className="font-medium">{topic.name}</span>
                    </div>
                    <div className="text-sm">
                      <div className="flex justify-between">
                        <span>Progress:</span>
                        <span>{topic.progress}%</span>
                      </div>
                      <ProgressBar progress={topic.progress} size="h-1.5" color={`bg-${index === 0 ? 'green' : index === 1 ? 'blue' : 'purple'}-500`} />
                    </div>
                  </div>
                  {topicDependencies[topic.id]?.length > 0 && (
                    <div className="absolute -left-4 top-1/2 transform -translate-y-1/2">
                      <div className="w-4 h-0.5 bg-gray-400"></div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </Card>

      {/* Recommendations */}
      <Card>
        <h3 className="text-xl font-semibold mb-4">Recommendations</h3>
        <div className="space-y-4">
          {course.topics.map(topic => {
            const mastery = getMasteryLevel(topic.progress);
            if (mastery.level !== 'Mastered') {
              return (
                <div key={topic.id} className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                  <div className="flex items-center gap-3 mb-2">
                    <Target className="w-5 h-5 text-blue-500" />
                    <h4 className="font-medium">{topic.name}</h4>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {mastery.level === 'Beginner' ? 'Start with basic concepts and practice exercises.' :
                     mastery.level === 'Developing' ? 'Focus on understanding key principles and solving more complex problems.' :
                     'Review advanced topics and work on challenging applications.'}
                  </p>
                </div>
              );
            }
            return null;
          })}
        </div>
      </Card>
    </div>
  );
};

export default ProgressView; 