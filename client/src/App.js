import React, { useState, useEffect } from 'react';
import { Home, PlusCircle, BookOpen, Target, HelpCircle, FileText, Lightbulb, Sparkles, CheckCircle, XCircle, ArrowLeft, ChevronRight, BarChart3, Edit3, Brain, ListChecks, Search, UploadCloud, Settings, User, Bell } from 'lucide-react';

// Mock Data
const initialCourses = [
  {
    id: 'cs101',
    name: 'Probabilistic AI',
    description: 'Explore the fundamentals of probabilistic reasoning in AI.',
    progress: 73,
    topics: [
      { id: 'topic1_1', name: 'Introduction to Probability', progress: 90, files: [{name: 'Lecture1.pdf', url:'#'}, {name:'Cheatsheet.png', url:'#'}], notes: 'Basic concepts of probability, sample spaces, events. Bayes\' theorem is crucial here.' },
      { id: 'topic1_2', name: 'Bayesian Networks', progress: 60, files: [{name: 'Chapter2_BN.pdf', url:'#'}], notes: 'Understanding conditional independence and graphical models.' },
      { id: 'topic1_3', name: 'Hidden Markov Models', progress: 45, files: [], notes: 'Focus on sequences and state transitions.' },
    ],
    color: 'bg-blue-500',
    icon: <Brain className="w-8 h-8 text-blue-100" />
  },
  {
    id: 'math202',
    name: 'Linear Algebra',
    description: 'Master vectors, matrices, and linear transformations.',
    progress: 45,
    topics: [
      { id: 'topic2_1', name: 'Vectors and Spaces', progress: 70, files: [], notes: 'Vector addition, scalar multiplication, dot product.' },
      { id: 'topic2_2', name: 'Matrix Operations', progress: 30, files: [], notes: 'Matrix multiplication, determinants, inverses.' },
    ],
    color: 'bg-green-500',
    icon: <ListChecks className="w-8 h-8 text-green-100" />
  },
];

const mockQuizQuestions = {
  cs101_topic1_1: [
    { id: 'q1', text: 'What is the definition of conditional probability?', type: 'recall', explanation: 'Conditional probability P(A|B) is the likelihood of event A occurring given that B is true. Formula: P(A|B) = P(A ∩ B) / P(B). It is foundational for Bayes\' Theorem.' },
    { id: 'q2', text: 'Explain Bayes\' Theorem and its components.', type: 'test', explanation: 'Bayes\' Theorem describes the probability of an event based on prior knowledge of conditions that might be related to the event. Formula: P(A|B) = [P(B|A) * P(A)] / P(B).' },
  ],
  cs101_topic1_2: [
    { id: 'q3', text: 'What is a d-separation in Bayesian Networks?', type: 'test', explanation: 'D-separation (direction-dependent separation) is a graphical criterion used to determine whether a set of nodes X is independent of another set Y, given a third set Z in a Bayesian network.' },
  ]
};

// Helper Components
const ProgressBar = ({ progress, size = 'h-2.5', color = 'bg-blue-600' }) => (
  <div className={`w-full bg-gray-200 rounded-full ${size} dark:bg-gray-700`}>
    <div className={`${color} ${size} rounded-full`} style={{ width: `${progress}%` }}></div>
  </div>
);

const CircularProgressBar = ({ progress, size = 120, strokeWidth = 10, color = "text-blue-600", trackColor = "text-gray-200" }) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (progress / 100) * circumference;

  return (
    <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
      <svg className="absolute w-full h-full transform -rotate-90">
        <circle
          className={trackColor}
          stroke="currentColor"
          strokeWidth={strokeWidth}
          fill="transparent"
          r={radius}
          cx={size / 2}
          cy={size / 2}
        />
        <circle
          className={color}
          stroke="currentColor"
          strokeWidth={strokeWidth}
          fill="transparent"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          r={radius}
          cx={size / 2}
          cy={size / 2}
        />
      </svg>
      <span className={`absolute text-xl font-bold ${color}`}>{progress}%</span>
    </div>
  );
};

const Card = ({ children, className = "" }) => (
  <div className={`bg-white dark:bg-gray-800 shadow-lg rounded-xl p-6 ${className}`}>
    {children}
  </div>
);

const Button = ({ children, onClick, variant = 'primary', size = 'md', className = '', icon: Icon, disabled = false }) => {
  const baseStyles = "font-medium rounded-lg focus:outline-none focus:ring-2 focus:ring-opacity-50 transition-all duration-150 ease-in-out flex items-center justify-center gap-2";
  const variantStyles = {
    primary: `bg-blue-600 hover:bg-blue-700 text-white focus:ring-blue-500 ${disabled ? 'bg-blue-300 hover:bg-blue-300 cursor-not-allowed' : ''}`,
    secondary: `bg-gray-200 hover:bg-gray-300 text-gray-800 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600 focus:ring-gray-400 ${disabled ? 'bg-gray-100 hover:bg-gray-100 cursor-not-allowed' : ''}`,
    danger: `bg-red-500 hover:bg-red-600 text-white focus:ring-red-400 ${disabled ? 'bg-red-300 hover:bg-red-300 cursor-not-allowed' : ''}`,
    ghost: `bg-transparent hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 focus:ring-gray-400 ${disabled ? 'text-gray-400 cursor-not-allowed' : ''}`,
  };
  const sizeStyles = {
    sm: "px-3 py-1.5 text-sm",
    md: "px-4 py-2 text-base",
    lg: "px-6 py-3 text-lg",
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`${baseStyles} ${variantStyles[variant]} ${sizeStyles[size]} ${className}`}
    >
      {Icon && <Icon size={size === 'sm' ? 16 : 20} />}
      {children}
    </button>
  );
};

// Main App Component
function App() {
  const [currentView, setCurrentView] = useState('dashboard'); // dashboard, addCourse, courseView, topicView, quizMode
  const [courses, setCourses] = useState(initialCourses);
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
        return selectedTopic && selectedCourse ? <TopicView topic={selectedTopic} courseName={selectedCourse.name} onNavigate={navigateTo} onStartQuiz={startQuiz} onPersonalizeNotes={handlePersonalizeNotes} feedbackMessage={feedbackMessage} /> : <p>Topic not found.</p>;
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

// Views
const DashboardView = ({ courses, onNavigate, onAddCourseClick, searchTerm, setSearchTerm }) => (
  <div>
    <div className="mb-6 flex flex-col sm:flex-row justify-between items-center gap-4">
        <h2 className="text-2xl font-semibold">Hey User, welcome back!</h2>
        <Button onClick={onAddCourseClick} icon={PlusCircle} size="md">Add New Course</Button>
    </div>
    
    <div className="mb-6">
        <div className="relative">
            <input 
                type="text"
                placeholder="Search courses..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full p-3 pl-10 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 focus:ring-2 focus:ring-blue-500 outline-none"
            />
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
        </div>
    </div>

    {courses.length === 0 && !searchTerm && (
        <Card className="text-center">
            <BookOpen size={48} className="mx-auto text-gray-400 mb-4" />
            <h3 className="text-xl font-semibold mb-2">No courses yet!</h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">Click "Add New Course" to get started with your learning journey.</p>
        </Card>
    )}

    {courses.length === 0 && searchTerm && (
        <Card className="text-center">
            <Search size={48} className="mx-auto text-gray-400 mb-4" />
            <h3 className="text-xl font-semibold mb-2">No courses found for "{searchTerm}"</h3>
            <p className="text-gray-600 dark:text-gray-400">Try a different search term or add a new course.</p>
        </Card>
    )}
    
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {courses.map(course => (
        <Card key={course.id} className={`hover:shadow-xl transition-shadow cursor-pointer ${course.color} text-white`}>
          <div onClick={() => onNavigate('courseView', { courseId: course.id })}>
            <div className="flex justify-between items-start mb-4">
                {course.icon || <BookOpen className="w-8 h-8 text-gray-100" />}
                <span className="text-xs font-semibold bg-white/20 px-2 py-1 rounded-full">{course.topics.length} Topics</span>
            </div>
            <h3 className="text-xl font-bold mb-2">{course.name}</h3>
            <p className="text-sm opacity-90 mb-4 h-10 overflow-hidden">{course.description}</p>
            <div className="flex items-center justify-between text-sm mb-1">
                <span>Overall Progress</span>
                <span>{course.progress}%</span>
            </div>
            <ProgressBar progress={course.progress} size="h-2" color="bg-white/50" />
          </div>
        </Card>
      ))}
    </div>
  </div>
);

const AddCourseView = ({ onAddCourse, onCancel }) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  // Mock file handling
  const [files, setFiles] = useState([]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!name.trim()) {
        alert("Course name is required.");
        return;
    }
    onAddCourse({ name, description, files }); // Pass files too, if you handle them
  };
  
  const handleFileChange = (e) => {
    if (e.target.files) {
        setFiles(Array.from(e.target.files).map(file => file.name)); // Store file names for now
    }
  };

  return (
    <Card className="max-w-2xl mx-auto">
      <h2 className="text-2xl font-semibold mb-6 text-center">Add New Course</h2>
      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label htmlFor="courseName" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Course Name <span className="text-red-500">*</span></label>
          <input
            type="text"
            id="courseName"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
            className="w-full p-3 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 focus:ring-2 focus:ring-blue-500 outline-none"
            placeholder="e.g., Introduction to Quantum Physics"
          />
        </div>
        <div>
          <label htmlFor="courseDescription" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Description</label>
          <textarea
            id="courseDescription"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            rows="3"
            className="w-full p-3 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 focus:ring-2 focus:ring-blue-500 outline-none"
            placeholder="A brief overview of the course content and objectives."
          />
        </div>
        <div>
            <label htmlFor="courseMaterials" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Course Materials (Optional)</label>
            <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 dark:border-gray-600 border-dashed rounded-md">
                <div className="space-y-1 text-center">
                    <UploadCloud className="mx-auto h-12 w-12 text-gray-400" />
                    <div className="flex text-sm text-gray-600 dark:text-gray-400">
                        <label htmlFor="file-upload" className="relative cursor-pointer bg-white dark:bg-gray-700 rounded-md font-medium text-blue-600 hover:text-blue-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-blue-500 dark:ring-offset-gray-800">
                            <span>Upload files</span>
                            <input id="file-upload" name="file-upload" type="file" className="sr-only" multiple onChange={handleFileChange} />
                        </label>
                        <p className="pl-1">or drag and drop</p>
                    </div>
                    <p className="text-xs text-gray-500 dark:text-gray-500">PDF, DOCX, PPTX, TXT up to 10MB</p>
                </div>
            </div>
            {files.length > 0 && (
                <div className="mt-2 text-sm text-gray-500 dark:text-gray-400">
                    Selected files: {files.join(', ')}
                </div>
            )}
        </div>
        <div className="flex justify-end gap-4 pt-4">
          <Button type="button" onClick={onCancel} variant="secondary">Cancel</Button>
          <Button type="submit" icon={PlusCircle}>Add Course</Button>
        </div>
      </form>
    </Card>
  );
};


const CourseView = ({ course, onNavigate, onStartQuiz }) => (
  <div className="space-y-8">
    <Card>
        <div className="flex flex-col md:flex-row items-center gap-6 md:gap-8">
            <CircularProgressBar progress={course.progress} size={160} strokeWidth={12} color={`text-${course.color.split('-')[1]}-600 dark:text-${course.color.split('-')[1]}-400`} />
            <div className="flex-1 text-center md:text-left">
                <h2 className="text-3xl font-bold mb-2">{course.name}</h2>
                <p className="text-gray-600 dark:text-gray-400 mb-4">{course.description || "No description available for this course."}</p>
                <div className="flex flex-wrap justify-center md:justify-start gap-3">
                    <Button onClick={() => onStartQuiz('personalized', course.id)} icon={Target} className="bg-green-500 hover:bg-green-600 text-white">Personalized Quiz</Button>
                    <Button onClick={() => alert('Full progress view (knowledge graph) coming soon!')} variant="secondary" icon={BarChart3}>View Full Progress</Button>
                    <Button onClick={() => alert('Quiz history coming soon!')} variant="secondary" icon={ListChecks}>Quiz History</Button>
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
                <div className="flex-1">
                    <h4 className="text-xl font-semibold text-blue-600 dark:text-blue-400 hover:underline cursor-pointer" onClick={() => onNavigate('topicView', { courseId: course.id, topicId: topic.id })}>{topic.name}</h4>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mt-1 truncate">{topic.notes || "No notes for this topic yet."}</p>
                </div>
                <div className="w-full sm:w-auto flex items-center gap-4 mt-3 sm:mt-0">
                    <div className="w-24 text-right">
                        <span className="text-sm font-medium">{topic.progress}%</span>
                        <ProgressBar progress={topic.progress} size="h-2" color={`bg-${course.color.split('-')[1]}-500`} />
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

const TopicView = ({ topic, courseName, onNavigate, onStartQuiz, onPersonalizeNotes, feedbackMessage }) => (
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


const QuizView = ({ question, userAnswer, setUserAnswer, showExplanation, onSubmitAnswer, onNextQuestion, isLastQuestion, quizType, courseName, topicName }) => {
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
      <p className="text-sm text-gray-500 dark:text-gray-400 text-center mb-6">Question {currentQuestionIndex + 1} of {currentQuizQuestions.length}</p>
      
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
            {/* Simulate linking to source material */}
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


// Global variables needed for QuizView to access (React context would be better for larger apps)
let selectedCourse, selectedTopic, currentQuestionIndex, currentQuizQuestions;

// This effect hook is a workaround to update global-like variables for QuizView.
// In a real app, manage this state properly via props drilling or context.
const AppWrapper = () => {
    const [appState, setAppState] = useState({
        _selectedCourse: null,
        _selectedTopic: null,
        _currentQuestionIndex: 0,
        _currentQuizQuestions: []
    });

    // This is a bit of a hack. Ideally, QuizView would get all its data via props.
    // Or use React Context.
    useEffect(() => {
        selectedCourse = appState._selectedCourse;
        selectedTopic = appState._selectedTopic;
        currentQuestionIndex = appState._currentQuestionIndex;
        currentQuizQuestions = appState._currentQuizQuestions;
    }, [appState]);

    // Pass a setter to App so it can update these "global" values
    // This is not standard React practice but done here for simplicity with the current structure.
    return <App setGlobalQuizState={setAppState} />;
}


export default App;

