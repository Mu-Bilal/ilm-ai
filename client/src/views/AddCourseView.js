import React, { useState } from 'react';
import { UploadCloud, PlusCircle, Link, Loader2 } from 'lucide-react';
import { Card, Button } from '../components/HelperComponents';

const AddCourseView = ({ onAddCourse, onCancel }) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [files, setFiles] = useState([]);
  const [urls, setUrls] = useState(['']);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!name.trim()) {
        setError("Course name is required.");
        return;
    }

    setIsLoading(true);
    setError('');

    try {
        // Filter out empty URLs
        const validUrls = urls.filter(url => url.trim() !== '');
        
        // Create request body
        const requestBody = {
            title: name,
            description: description,
            urls: validUrls
        };

        // Debug request body
        console.log('Request body:', requestBody);

        // Send to backend
        const response = await fetch('http://localhost:8000/api/generate-course', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to create course');
        }

        const data = await response.json();
        onAddCourse(data);
    } catch (err) {
        setError(err.message || 'Failed to create course. Please try again.');
    } finally {
        setIsLoading(false);
    }
  };
  
  const handleFileChange = (e) => {
    if (e.target.files) {
        setFiles(Array.from(e.target.files));
    }
  };

  const handleUrlChange = (index, value) => {
    const newUrls = [...urls];
    newUrls[index] = value;
    setUrls(newUrls);
  };

  const addUrlField = () => {
    setUrls([...urls, '']);
  };

  const removeUrlField = (index) => {
    const newUrls = urls.filter((_, i) => i !== index);
    setUrls(newUrls);
  };

  return (
    <Card className="max-w-2xl mx-auto">
      <h2 className="text-2xl font-semibold mb-6 text-center">Add New Course</h2>
      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 text-red-600 rounded-lg">
          {error}
        </div>
      )}
      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label htmlFor="courseName" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Course Name <span className="text-red-500">*</span></label>
          <input
            type="text"
            id="courseName"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
            disabled={isLoading}
            className="w-full p-3 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 focus:ring-2 focus:ring-blue-500 outline-none disabled:opacity-50"
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
            disabled={isLoading}
            className="w-full p-3 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 focus:ring-2 focus:ring-blue-500 outline-none disabled:opacity-50"
            placeholder="A brief overview of the course content and objectives."
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Course Materials URLs</label>
          <div className="space-y-2">
            {urls.map((url, index) => (
              <div key={index} className="flex gap-2">
                <input
                  type="url"
                  value={url}
                  onChange={(e) => handleUrlChange(index, e.target.value)}
                  disabled={isLoading}
                  className="flex-1 p-3 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 focus:ring-2 focus:ring-blue-500 outline-none disabled:opacity-50"
                  placeholder="https://example.com/course-material"
                />
                {urls.length > 1 && (
                  <button
                    type="button"
                    onClick={() => removeUrlField(index)}
                    disabled={isLoading}
                    className="p-3 text-red-500 hover:text-red-600 disabled:opacity-50"
                  >
                    Remove
                  </button>
                )}
              </div>
            ))}
            <button
              type="button"
              onClick={addUrlField}
              disabled={isLoading}
              className="flex items-center gap-2 text-blue-500 hover:text-blue-600 disabled:opacity-50"
            >
              <Link className="w-4 h-4" />
              Add another URL
            </button>
          </div>
        </div>
        <div>
            <label htmlFor="courseMaterials" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Course Materials (Optional)</label>
            <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 dark:border-gray-600 border-dashed rounded-md">
                <div className="space-y-1 text-center">
                    <UploadCloud className="mx-auto h-12 w-12 text-gray-400" />
                    <div className="flex text-sm text-gray-600 dark:text-gray-400">
                        <label htmlFor="file-upload" className="relative cursor-pointer bg-white dark:bg-gray-700 rounded-md font-medium text-blue-600 hover:text-blue-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-blue-500 dark:ring-offset-gray-800">
                            <span>Upload files</span>
                            <input 
                                id="file-upload" 
                                name="file-upload" 
                                type="file" 
                                className="sr-only" 
                                multiple 
                                onChange={handleFileChange}
                                disabled={isLoading}
                            />
                        </label>
                        <p className="pl-1">or drag and drop</p>
                    </div>
                    <p className="text-xs text-gray-500 dark:text-gray-500">PDF, DOCX, PPTX, TXT up to 10MB</p>
                </div>
            </div>
            {files.length > 0 && (
                <div className="mt-2 text-sm text-gray-500 dark:text-gray-400">
                    Selected files: {files.map(file => file.name).join(', ')}
                </div>
            )}
        </div>
        <div className="flex justify-end gap-4 pt-4">
          <Button type="button" onClick={onCancel} variant="secondary" disabled={isLoading}>Cancel</Button>
          <Button type="submit" icon={isLoading ? Loader2 : PlusCircle} disabled={isLoading} className="min-w-[120px]">
            {isLoading ? 'Creating...' : 'Add Course'}
          </Button>
        </div>
      </form>
    </Card>
  );
};

export default AddCourseView; 