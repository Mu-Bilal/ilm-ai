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

// Main App Component
function App() {
  const [currentView, setCurrentView] = useState('dashboard'); // dashboard, addCourse, courseView, topicView, quizMode
  const [courses, setCourses] = useState(initialCourses);
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
  const handleAddCourse = (newCourseData) => {
    const newCourse = {
      id: `course${Date.now()}`,
      progress: 0,
      topics: [],
      ...newCourseData,
      color: ['bg-purple-500', 'bg-pink-500', 'bg-indigo-500'][courses.length % 3], // Cycle through some colors
      icon: [<BookOpen className="w-8 h-8 text-purple-100" />, <Target className="w-8 h-8 text-pink-100" />, <Lightbulb className="w-8 h-8 text-indigo-100" />][courses.length % 3]
    };
    setCourses([...courses, newCourse]);
    navigateTo('dashboard');
  };

  // Quiz Logic
  const startQuiz = (type, courseId, topicId = null) => {
    setQuizType(type);
    let questions = [];
    if (topicId) {
      questions = mockQuizQuestions[`${courseId}_${topicId}`] || [];
      if (type === 'flash') questions = questions.filter(q => q.type === 'recall');
    } else { // Course-level personalized quiz
      // Mock personalized quiz: take first question from each topic for simplicity
      const course = courses.find(c => c.id === courseId);
      course?.topics.forEach(topic => {
        const topicQuestions = mockQuizQuestions[`${courseId}_${topic.id}`];
        if (topicQuestions && topicQuestions.length > 0) {
          questions.push(topicQuestions[0]); 
        }
      });
    }
    
    if (questions.length === 0) {
        // Add a default question if none are found to prevent crashing
        questions.push({ id: 'q_default', text: 'No questions available for this section yet. Try another one!', explanation: 'Please add questions to the mockQuizQuestions object for this topic/course.'});
    }

    setCurrentQuizQuestions(questions);
    setCurrentQuestionIndex(0);
    navigateTo('quizMode', { courseId, topicId });
  };

  const handleSubmitAnswer = () => {
    setShowExplanation(true);
    // In a real app, you'd check the answer and update progress
  };

  const handleNextQuestion = () => {
    if (currentQuestionIndex < currentQuizQuestions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
      setUserAnswer('');
      setShowExplanation(false);
    } else {
      // Quiz finished
      if (selectedTopic) navigateTo('topicView', { courseId: selectedCourse.id, topicId: selectedTopic.id });
      else if (selectedCourse) navigateTo('courseView', { courseId: selectedCourse.id });
      else navigateTo('dashboard');
    }
  };
  
  const handlePersonalizeNotes = (topic) => {
    setFeedbackMessage(`Personalized notes for "${topic.name}" are being generated based on your weakest points... (This is a simulation)`);
    // In a real app, this would trigger a backend process
    // For now, we can just show a message and maybe update the notes after a delay
    setTimeout(() => {
        const updatedCourses = courses.map(c => {
            if (c.id === selectedCourse.id) {
                return {
                    ...c,
                    topics: c.topics.map(t => {
                        if (t.id === topic.id) {
                            return {...t, notes: `(Personalized) ${t.notes} \n\nKey areas to focus: [Simulated weak point 1], [Simulated weak point 2].` };
                        }
                        return t;
                    })
                };
            }
            return c;
        });
        setCourses(updatedCourses);
        const updatedCourse = updatedCourses.find(c => c.id === selectedCourse.id);
        setSelectedCourse(updatedCourse);
        setSelectedTopic(updatedCourse?.topics.find(t => t.id === topic.id));
        setFeedbackMessage(`Personalized notes for "${topic.name}" have been updated!`);
    }, 2000);
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
            <h1 className="text-3xl font-bold text-blue-600 dark:text-blue-400">LearnSmart AI</h1>
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

