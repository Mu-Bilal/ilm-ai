import React from 'react';
import { BookOpen, PlusCircle, Search } from 'lucide-react';
import { Card, Button, ProgressBar } from '../components/HelperComponents';

const DashboardView = ({ courses, onNavigate, onAddCourseClick, searchTerm, setSearchTerm }) => {
  console.log('DashboardView courses:', courses); // Debug log
  
  return (
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
        <Card key={course.id} className={`hover:shadow-xl transition-shadow cursor-pointer ${course.color} text-black`}>
          <div onClick={() => onNavigate('courseView', { courseId: course.id })}>
            <div className="flex justify-between items-start mb-4">
                {course.icon || <BookOpen className="w-8 h-8 text-gray-300" />}
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
};

export default DashboardView; 