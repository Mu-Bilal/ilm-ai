import aiohttp
from bs4 import BeautifulSoup
from typing import List, Optional, Dict
import asyncio
import os
from pathlib import Path

class CourseGenerator:
    def __init__(self):
        self.download_dir = Path("downloads")
        self.download_dir.mkdir(exist_ok=True)

    async def generate_course(self, title: str, description: str, urls: List[str], topics: Optional[List[str]] = None) -> Dict:
        """
        Generate a course from the provided URLs and optional topics.
        """
        # Process URLs and download content
        downloaded_files = await self._process_urls(urls)
        
        # Generate course structure
        course = {
            "title": title,
            "description": description,
            "topics": topics or [],
            "files": downloaded_files,
            "progress": 0
        }
        
        return course

    async def _process_urls(self, urls: List[str]) -> List[Dict]:
        """
        Process URLs and download relevant files.
        """
        downloaded_files = []
        
        async with aiohttp.ClientSession() as session:
            tasks = [self._process_single_url(session, url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    print(f"Error processing URL: {str(result)}")
                    continue
                downloaded_files.extend(result)
        
        return downloaded_files

    async def _process_single_url(self, session: aiohttp.ClientSession, url: str) -> List[Dict]:
        """
        Process a single URL and download relevant files.
        """
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to fetch URL: {url}")
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Find and download PDFs, documents, etc.
                files = []
                for link in soup.find_all('a'):
                    href = link.get('href')
                    if href and self._is_downloadable_file(href):
                        file_info = await self._download_file(session, href, url)
                        if file_info:
                            files.append(file_info)
                
                return files
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            return []

    def _is_downloadable_file(self, url: str) -> bool:
        """
        Check if the URL points to a downloadable file.
        """
        downloadable_extensions = ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.txt']
        return any(url.lower().endswith(ext) for ext in downloadable_extensions)

    async def _download_file(self, session: aiohttp.ClientSession, file_url: str, base_url: str) -> Optional[Dict]:
        """
        Download a file and return its information.
        """
        try:
            # Handle relative URLs
            if not file_url.startswith(('http://', 'https://')):
                file_url = f"{base_url.rstrip('/')}/{file_url.lstrip('/')}"
            
            async with session.get(file_url) as response:
                if response.status != 200:
                    return None
                
                # Get filename from URL or Content-Disposition header
                filename = file_url.split('/')[-1]
                content_disposition = response.headers.get('Content-Disposition')
                if content_disposition:
                    import re
                    match = re.search(r'filename="(.+?)"', content_disposition)
                    if match:
                        filename = match.group(1)
                
                # Save file
                file_path = self.download_dir / filename
                with open(file_path, 'wb') as f:
                    f.write(await response.read())
                
                return {
                    "name": filename,
                    "url": str(file_path),
                    "type": self._get_file_type(filename)
                }
        except Exception as e:
            print(f"Error downloading file {file_url}: {str(e)}")
            return None

    def _get_file_type(self, filename: str) -> str:
        """
        Determine the type of file based on its extension.
        """
        ext = filename.lower().split('.')[-1]
        if ext in ['pdf']:
            return 'document'
        elif ext in ['doc', 'docx']:
            return 'word'
        elif ext in ['ppt', 'pptx']:
            return 'presentation'
        elif ext in ['txt']:
            return 'text'
        return 'other' 