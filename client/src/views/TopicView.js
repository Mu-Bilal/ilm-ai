import React from 'react';
import { Lightbulb, HelpCircle, ListChecks, Sparkles, FileText } from 'lucide-react';
import { Card, Button, CircularProgressBar } from '../components/HelperComponents';

const TopicView = ({ 
  topic, 
  courseName, 
  onNavigate, 
  onStartQuiz, 
  onPersonalizeNotes, 
  feedbackMessage,
  selectedCourse 
}) => (
    <div className="space-y-8">
        <Card>
            <div className="flex flex-col md:flex-row items-center gap-6 md:gap-8">
                <CircularProgressBar progress={topic.progress} size={140} strokeWidth={10} color="text-green-600 dark:text-green-400" />
                <div className="flex-1 text-center md:text-left">
                    <p className="text-sm text-blue-500 dark:text-blue-400 font-medium mb-1">{courseName}</p>
                    <h2 className="text-3xl font-bold mb-3">{topic.name}</h2>
                    <div className="flex flex-wrap justify-center md:justify-start gap-3">
                        <Button onClick={() => onStartQuiz('flash', selectedCourse.id, topic.id)} icon={Lightbulb} className="bg-yellow-500 hover:bg-yellow-600 text-white">Flash Quiz</Button>
                        <Button onClick={() => onStartQuiz('test', selectedCourse.id, topic.id)} icon={HelpCircle} className="bg-purple-500 hover:bg-purple-600 text-white">Test Quiz</Button>
                        <Button onClick={() => alert('Topic-specific quiz history coming soon!')} variant="secondary" icon={ListChecks}>Quiz History</Button>
                    </div>
                </div>
            </div>
        </Card>

        {feedbackMessage && (
            <Card className="bg-green-50 border border-green-300 dark:bg-green-900 dark:border-green-700">
                <p className="text-green-700 dark:text-green-200 font-medium">{feedbackMessage}</p>
            </Card>
        )}

        <Card>
            <h3 className="text-xl font-semibold mb-3">Notes</h3>
            <Button onClick={() => onPersonalizeNotes(topic)} icon={Sparkles} size="sm" className="mb-4 float-right -mt-2">Personalize Notes</Button>
            <div className="prose dark:prose-invert max-w-none bg-gray-50 dark:bg-gray-700 p-4 rounded-md min-h-[100px]">
                {topic.notes ? <p>{topic.notes.split('\n').map((line, i) => <span key={i}>{line}<br/></span>)}</p> : <p className="text-gray-500">No notes available for this topic yet.</p>}
            </div>
        </Card>

        <Card>
            <h3 className="text-xl font-semibold mb-3">Files & Materials</h3>
            {topic.files && topic.files.length > 0 ? (
                <ul className="space-y-2">
                {topic.files.map((file, index) => (
                    <li key={index} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-md hover:bg-gray-100 dark:hover:bg-gray-600">
                        <div className="flex items-center gap-3">
                            <FileText className="w-5 h-5 text-blue-500" />
                            <span>{file.name}</span>
                        </div>
                        <a href={file.url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 font-medium">
                            View
                        </a>
                    </li>
                ))}
                </ul>
            ) : (
                <p className="text-gray-500 dark:text-gray-400">No files uploaded for this topic.</p>
            )}
        </Card>
    </div>
);

export default TopicView; 