import React, { useState } from 'react';
import { UploadCloud, PlusCircle } from 'lucide-react';
import { Card, Button } from '../components/HelperComponents';

const AddCourseView = ({ onAddCourse, onCancel }) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [files, setFiles] = useState([]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!name.trim()) {
        alert("Course name is required.");
        return;
    }
    onAddCourse({ name, description, files });
  };
  
  const handleFileChange = (e) => {
    if (e.target.files) {
        setFiles(Array.from(e.target.files).map(file => file.name));
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

export default AddCourseView; 