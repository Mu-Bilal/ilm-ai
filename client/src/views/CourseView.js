import React from 'react';
import { Target, BarChart3, ListChecks, ChevronRight } from 'lucide-react';
import { Card, Button, ProgressBar, CircularProgressBar } from '../components/HelperComponents';

const CourseView = ({ course, onNavigate, onStartQuiz }) => (
  <div className="space-y-8">
    <Card>
        <div className="flex flex-col md:flex-row items-center gap-6 md:gap-8">
            <CircularProgressBar 
                progress={course.progress} 
                size={160} 
                strokeWidth={12}
            />
            <div className="flex-1 text-center md:text-left">
                <h2 className="text-3xl font-bold mb-2">{course.name}</h2>
                <p className="text-gray-600 dark:text-gray-400 mb-4">{course.description || "No description available for this course."}</p>
                <div className="flex flex-wrap justify-center md:justify-start gap-3">
                    <Button onClick={() => onStartQuiz('personalized', course.id)} icon={Target} className="bg-green-500 hover:bg-green-600 text-white">Personalized Quiz</Button>
                    <Button onClick={() => onNavigate('quizHistory', { courseId: course.id })} variant="secondary" icon={ListChecks}>Quiz History</Button>
                </div>
            </div>
        </div>
    </Card>

    <div>
        <h3 className="text-2xl font-semibold mb-4">Topics</h3>
        {course.topics.length === 0 ? (
            <Card><p className="text-gray-600 dark:text-gray-400">No topics added to this course yet. You can add topics by editing the course (feature coming soon!).</p></Card>
        ) : (
        <div className="space-y-4">
            {course.topics.map(topic => (
            <Card key={topic.id} className="hover:shadow-md transition-shadow">
                <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                <div className="flex-1 min-w-0">
                    <h4 className="text-xl font-semibold text-blue-600 dark:text-blue-400 hover:underline cursor-pointer truncate" onClick={() => onNavigate('topicView', { courseId: course.id, topicId: topic.id })}>{topic.name}</h4>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mt-1 line-clamp-2">{topic.notes || "No notes for this topic yet."}</p>
                </div>
                <div className="w-full sm:w-auto flex items-center gap-4 mt-3 sm:mt-0">
                    <div className="w-24 text-right">
                        <span className="text-sm font-medium">{topic.progress}%</span>
                        <ProgressBar 
                            progress={topic.progress} 
                            size="h-2"
                        />
                    </div>
                    <Button onClick={() => onNavigate('topicView', { courseId: course.id, topicId: topic.id })} variant="ghost" size="sm" icon={ChevronRight} className="p-2">
                        <span className="sr-only">View Topic</span>
                    </Button>
                </div>
                </div>
            </Card>
            ))}
        </div>
        )}
    </div>
  </div>
);

export default CourseView; 