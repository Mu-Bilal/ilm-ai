import React, { useState, useEffect } from 'react';
import { Home, PlusCircle, BookOpen, Target, HelpCircle, FileText, Lightbulb, Sparkles, CheckCircle, XCircle, ArrowLeft, ChevronRight, BarChart3, Edit3, Brain, ListChecks, Search, UploadCloud, Settings, User, Bell } from 'lucide-react';
import { ProgressBar, CircularProgressBar, Card, Button } from './components/HelperComponents';

// Import mock data
import { initialCourses, mockQuizQuestions } from './mockData';

// Import views
import DashboardView from './views/DashboardView';
import AddCourseView from './views/AddCourseView';
import CourseView from './views/CourseView';
import TopicView from './views/TopicView';
import QuizView from './views/QuizView';
import ProgressView from './views/ProgressView';
import QuizHistoryView from './views/QuizHistoryView';

// Answer checking function
const checkAnswer = (userAnswer, correctAnswer) => {
  // Convert both strings to lowercase and remove extra whitespace
  const normalize = (str) => str.toLowerCase().trim().replace(/\s+/g, ' ');
  const user = normalize(userAnswer);
  const correct = normalize(correctAnswer);

  // If strings are exactly the same, return 1.0
  if (user === correct) return 1.0;

  // Split into words
  const userWords = new Set(user.split(' '));
  const correctWords = new Set(correct.split(' '));

  // Calculate Jaccard similarity
  const intersection = new Set([...userWords].filter(x => correctWords.has(x)));
  const union = new Set([...userWords, ...correctWords]);
  
  const similarity = intersection.size / union.size;

  // Consider it correct if similarity is above 0.7
  return similarity >= 0.7;
};

// Hardcoded course progress
const hardcodedProgress = {
  'cs101': {
    overall: 75,
    topics: {
      '1 Fundamentals': 85,
      '2 Bayesian Linear Regression': 65,
      '3 Kalman Filters': 75,
      '4 Gaussian Processes': 70,
      '5 Variational Inference': 60,
      '6 Markov Chain Monte Carlo Methods': 55,
      '7 Bayesian Deep Learning': 50,
      '8 Active Learning': 65,
      '9 Bayesian Optimization': 70,
      '10 Markov Decision Processes': 60,
      '11 Tabular Reinforcement Learning': 55,
      '12 Model-free Approximate Reinforcement Learning': 50,
      '13 Model-based Approximate Reinforcement Learning': 45
    }
  }
};

// Main App Component
function App() {
  const [currentView, setCurrentView] = useState('dashboard'); // dashboard, addCourse, courseView, topicView, quizMode, progressView, quizHistory
  const [courses, setCourses] = useState(initialCourses.map(course => {
    if (hardcodedProgress[course.id]) {
      return {
        ...course,
        progress: hardcodedProgress[course.id].overall,
        topics: course.topics.map(topic => ({
          ...topic,
          progress: hardcodedProgress[course.id].topics[topic.id] || topic.progress
        }))
      };
    }
    return course;
  }));
  console.log('App initialCourses:', initialCourses); // Debug log
  console.log('App courses state:', courses); // Debug log
  const [selectedCourse, setSelectedCourse] = useState(null);
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [quizType, setQuizType] = useState(''); // 'personalized', 'flash', 'test'
  const [currentQuizQuestions, setCurrentQuizQuestions] = useState([]);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [userAnswer, setUserAnswer] = useState('');
  const [showExplanation, setShowExplanation] = useState(false);
  const [feedbackMessage, setFeedbackMessage] = useState(''); // For "Personalized Notes Generated" etc.
  const [searchTerm, setSearchTerm] = useState('');
  const [isAnswerCorrect, setIsAnswerCorrect] = useState(null);
  const [quizHistory, setQuizHistory] = useState([]);
  const [quizStartTime, setQuizStartTime] = useState(null);
  const [correctAnswers, setCorrectAnswers] = useState(0);

  // Navigation
  const navigateTo = (view, params = {}) => {
    setCurrentView(view);
    if (params.courseId) {
      const course = courses.find(c => c.id === params.courseId);
      setSelectedCourse(course);
      if (params.topicId) {
        const topic = course?.topics.find(t => t.id === params.topicId);
        setSelectedTopic(topic);
      } else {
        setSelectedTopic(null);
      }
    } else {
        setSelectedCourse(null);
        setSelectedTopic(null);
    }
    setShowExplanation(false);
    setUserAnswer('');
    setFeedbackMessage('');
  };

  // Course Management
  const handleAddCourse = (courseData) => {
    const newCourse = {
      id: courseData.id || `course${Date.now()}`,
      progress: courseData.progress || 0,
      topics: courseData.topics || [],
      name: courseData.name,
      description: courseData.description,
      files: courseData.files || [],
      color: ['bg-purple-500', 'bg-pink-500', 'bg-indigo-500'][courses.length % 3],
      icon: [<BookOpen className="w-8 h-8 text-purple-100" />, <Target className="w-8 h-8 text-pink-100" />, <Lightbulb className="w-8 h-8 text-indigo-100" />][courses.length % 3]
    };
    setCourses([...courses, newCourse]);
    navigateTo('dashboard');
  };

  // Quiz Logic
  const startQuiz = (type, courseId, topicId = null) => {
    setQuizType(type);
    setQuizStartTime(Date.now());
    setCorrectAnswers(0);
    let questions = [];
    if (topicId) {
      questions = mockQuizQuestions[`${courseId}_${topicId}`] || [];
      if (type === 'flash') questions = questions.filter(q => q.type === 'recall');
    } else {
      const course = courses.find(c => c.id === courseId);
      course?.topics.forEach(topic => {
        const topicQuestions = mockQuizQuestions[`${courseId}_${topic.id}`];
        if (topicQuestions && topicQuestions.length > 0) {
          questions.push(topicQuestions[0]);
        }
      });
    }
    
    if (questions.length === 0) {
      questions.push({ id: 'q_default', text: 'No questions available for this section yet. Try another one!', explanation: 'Please add questions to the mockQuizQuestions object for this topic/course.'});
    }

    setCurrentQuizQuestions(questions);
    setCurrentQuestionIndex(0);
    navigateTo('quizMode', { courseId, topicId });
  };

  const handleSubmitAnswer = () => {
    const currentQuestion = currentQuizQuestions[currentQuestionIndex];
    const isCorrect = checkAnswer(userAnswer, currentQuestion.explanation);
    setIsAnswerCorrect(isCorrect);
    setShowExplanation(true);

    if (isCorrect) {
      setCorrectAnswers(prev => prev + 1);
    }

    // Update progress if answer is correct
    if (isCorrect && selectedCourse) {
      const updatedCourses = courses.map(course => {
        if (course.id === selectedCourse.id) {
          const updatedTopics = course.topics.map(topic => {
            if (selectedTopic && topic.id === selectedTopic.id) {
              // Calculate progress increase based on question type and current progress
              let progressIncrease;
              if (quizType === 'test') {
                // Test questions have higher weight
                progressIncrease = 10;
              } else if (quizType === 'flash') {
                // Flash questions have medium weight
                progressIncrease = 7;
              } else {
                // Personalized questions have lower weight
                progressIncrease = 5;
              }

              // Adjust progress increase based on current progress
              // Harder to gain progress when already at higher levels
              if (topic.progress >= 80) {
                progressIncrease *= 0.5;
              } else if (topic.progress >= 60) {
                progressIncrease *= 0.7;
              } else if (topic.progress >= 40) {
                progressIncrease *= 0.85;
              }

              // Calculate new progress
              const newProgress = Math.min(Math.round(topic.progress + progressIncrease), 100);
              return { ...topic, progress: newProgress };
            }
            return topic;
          });

          // Calculate new course progress based on weighted average of topics
          const totalWeight = updatedTopics.reduce((sum, topic) => sum + (topic.progress / 100), 0);
          const newCourseProgress = Math.round((totalWeight / updatedTopics.length) * 100);

          return {
            ...course,
            topics: updatedTopics,
            progress: newCourseProgress
          };
        }
        return course;
      });

      setCourses(updatedCourses);
      if (selectedCourse) {
        setSelectedCourse(updatedCourses.find(c => c.id === selectedCourse.id));
      }
    }
  };

  const handleNextQuestion = () => {
    if (currentQuestionIndex < currentQuizQuestions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
      setUserAnswer('');
      setShowExplanation(false);
    } else {
      // Quiz finished - record history
      const duration = Math.round((Date.now() - quizStartTime) / 60000); // Convert to minutes
      const score = Math.round((correctAnswers / currentQuizQuestions.length) * 100);
      
      const newQuizAttempt = {
        type: quizType,
        courseId: selectedCourse.id,
        courseName: selectedCourse.name,
        topicId: selectedTopic?.id,
        topicName: selectedTopic?.name,
        date: new Date().toISOString(),
        duration,
        score,
        correctAnswers,
        totalQuestions: currentQuizQuestions.length
      };

      setQuizHistory(prev => [newQuizAttempt, ...prev]);

      // Navigate back
      if (selectedTopic) navigateTo('topicView', { courseId: selectedCourse.id, topicId: selectedTopic.id });
      else if (selectedCourse) navigateTo('courseView', { courseId: selectedCourse.id });
      else navigateTo('dashboard');
    }
  };
  
  const handlePersonalizeNotes = async (topic) => {
    setFeedbackMessage(`Personalized notes for "${topic.name}" are being generated based on your weakest points...`);
    try {
        const response = await fetch('http://localhost:8000/api/personalize-notes', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_id: '1', // TODO: Replace with actual user ID
                course_id: selectedCourse.id,
                chapter_id: topic.name
            })
        });

        if (!response.ok) {
            throw new Error('Failed to generate personalized notes');
        }

        const data = await response.json();
        setFeedbackMessage(data.notes);
    } catch (error) {
        console.error('Error generating personalized notes:', error);
        setFeedbackMessage('Failed to generate personalized notes. Please try again.');
    }
  };

  // Filtered courses for dashboard
  const filteredCourses = courses.filter(course => 
    course.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Render current view
  const renderView = () => {
    if (feedbackMessage) {
        setTimeout(() => setFeedbackMessage(''), 3000); // Clear message after 3s
    }

    switch (currentView) {
      case 'dashboard':
        return <DashboardView courses={filteredCourses} onNavigate={navigateTo} onAddCourseClick={() => navigateTo('addCourse')} searchTerm={searchTerm} setSearchTerm={setSearchTerm} />;
      case 'addCourse':
        return <AddCourseView onAddCourse={handleAddCourse} onCancel={() => navigateTo('dashboard')} />;
      case 'courseView':
        return selectedCourse ? <CourseView course={selectedCourse} onNavigate={navigateTo} onStartQuiz={startQuiz} /> : <p>Course not found.</p>;
      case 'topicView':
        return selectedTopic && selectedCourse ? <TopicView 
          topic={selectedTopic} 
          courseName={selectedCourse.name} 
          onNavigate={navigateTo} 
          onStartQuiz={startQuiz} 
          onPersonalizeNotes={handlePersonalizeNotes} 
          feedbackMessage={feedbackMessage}
          selectedCourse={selectedCourse}
        /> : <p>Topic not found.</p>;
      case 'progressView':
        return selectedCourse ? <ProgressView course={selectedCourse} /> : <p>Course not found.</p>;
      case 'quizHistory':
        return selectedCourse ? <QuizHistoryView 
          course={selectedCourse} 
          quizHistory={quizHistory.filter(attempt => attempt.courseId === selectedCourse.id)}
          onNavigate={navigateTo}
        /> : <p>Course not found.</p>;
      case 'quizMode':
        const question = currentQuizQuestions[currentQuestionIndex];
        return question ? <QuizView
          question={question}
          userAnswer={userAnswer}
          setUserAnswer={setUserAnswer}
          showExplanation={showExplanation}
          onSubmitAnswer={handleSubmitAnswer}
          onNextQuestion={handleNextQuestion}
          isLastQuestion={currentQuestionIndex === currentQuizQuestions.length - 1}
          quizType={quizType}
          courseName={selectedCourse?.name}
          topicName={selectedTopic?.name}
          currentQuestionIndex={currentQuestionIndex}
          totalQuestions={currentQuizQuestions.length}
          isAnswerCorrect={isAnswerCorrect}
        /> : <p>Loading quiz...</p>;
      default:
        return <DashboardView courses={filteredCourses} onNavigate={navigateTo} onAddCourseClick={() => navigateTo('addCourse')} searchTerm={searchTerm} setSearchTerm={setSearchTerm} />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 dark:bg-gray-900 text-gray-900 dark:text-gray-100 font-inter p-4 sm:p-6 lg:p-8">
      <header className="mb-6 flex justify-between items-center">
        <div className="flex items-center gap-3">
            {currentView !== 'dashboard' && (
                <Button onClick={() => {
                    if (currentView === 'quizMode') {
                        if (selectedTopic) navigateTo('topicView', { courseId: selectedCourse.id, topicId: selectedTopic.id });
                        else if (selectedCourse) navigateTo('courseView', { courseId: selectedCourse.id });
                        else navigateTo('dashboard');
                    } else if (currentView === 'topicView' && selectedCourse) {
                        navigateTo('courseView', { courseId: selectedCourse.id });
                    } else if (currentView === 'courseView' || currentView === 'addCourse') {
                        navigateTo('dashboard');
                    } else {
                        navigateTo('dashboard');
                    }
                }} variant="ghost" size="sm" icon={ArrowLeft}>Back</Button>
            )}
            <h1 className="text-3xl font-bold text-blue-600 dark:text-blue-400">ILM AI</h1>
        </div>
        <div className="flex items-center gap-3">
            <Button variant="ghost" icon={Bell} className="relative">
                <span className="absolute top-0 right-0 block h-2 w-2 rounded-full bg-red-500 ring-2 ring-white dark:ring-gray-900"></span>
            </Button>
            <Button variant="ghost" icon={Settings} />
            <User className="w-8 h-8 text-gray-500 dark:text-gray-400" />
        </div>
      </header>
      <main>{renderView()}</main>
      <footer className="text-center mt-12 text-sm text-gray-500 dark:text-gray-400">
        <p>&copy; {new Date().getFullYear()} LearnSmart AI. Personalized learning journey.</p>
      </footer>
    </div>
  );
}

export default App;

