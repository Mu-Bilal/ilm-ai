import aiohttp
from bs4 import BeautifulSoup
from typing import List, Optional, Dict
import asyncio
import os
from pathlib import Path
from urllib.parse import urlparse
from app.services.linkAgent import get_topics

class CourseGenerator:
    def __init__(self):
        self.download_dir = Path("downloads")
        self.download_dir.mkdir(exist_ok=True)
        self.session_token = None

    def set_session_token(self, token: str):
        """
        Set the session token for authenticated requests.
        """
        self.session_token = token
        print("Session token set successfully")

    def _get_auth_headers(self) -> dict:
        """
        Get headers with authentication token if available.
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        if self.session_token:
            headers['Authorization'] = f'Bearer {self.session_token}'
            # You might also need to set a cookie
            headers['Cookie'] = f'session={self.session_token}'
            
        return headers

    async def generate_course(self, title: str, description: str, urls: List[str], topics: Optional[List[str]] = None) -> Dict:
        """
        Generate a course from the provided URLs and optional topics.
        """
        # Process URLs and download content
        downloaded_files = await self._process_urls(urls)
        # get topics from the downloaded files
        topics = []
        for file in downloaded_files:
            file_topics = await get_topics(file['url'])
            topics.extend(file_topics)

        # downloaded_files = []
        
        # Generate course structure
        course = {
            "name": title,
            "description": description,
            "topics": topics or [],
            "files": [],
            "progress": 0
        }
        
        return course

    async def _process_urls(self, urls: List[str]) -> List[Dict]:
        """
        Process URLs and download relevant files.
        """
        downloaded_files = []
        auth = aiohttp.BasicAuth("ml", "predict")
        async with aiohttp.ClientSession(auth=auth) as session:
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
            base_domain = self._get_base_domain(url)
            print(f"Processing URL: {url}")
            print(f"Base domain: {base_domain}")
            
            headers = self._get_auth_headers()
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    raise Exception(f"Failed to fetch URL: {url}")
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Find and download PDFs, documents, etc.
                files = []
                for link in soup.find_all('a'):
                    if len(files) >= 2:
                        break
                    href = link.get('href')
                    if href and self._is_downloadable_file(href) and self.is_course_file(href):
                        # Use base domain for relative URLs
                        file_info = await self._download_file(session, href, base_domain)
                        if file_info:
                            files.append(file_info)
                        
                
                return files
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            return []

    def is_course_file(self, url: str) -> bool:
        """
        Check if the URL points to a course file.
        """
        return (not "exam" in url.lower())

    def _is_downloadable_file(self, url: str) -> bool:
        """
        Check if the URL points to a downloadable file.
        """
        downloadable_extensions = ['.pdf']
        return any(url.lower().endswith(ext) for ext in downloadable_extensions)

    async def _download_file(self, session: aiohttp.ClientSession, file_url: str, base_url: str) -> Optional[Dict]:
        """
        Download a file and return its information.
        """
        try:
            # Handle relative URLs
            if not file_url.startswith(('http://', 'https://')):
                file_url = f"{base_url.rstrip('/')}/{file_url.lstrip('/')}"
            
            print(f"Attempting to download file from: {file_url}")
            headers = self._get_auth_headers()
            payload = {
                "username": "ml",
                "password": "predict",
            }
            async with session.post("https://las.inf.ethz.ch/", headers=headers, data=payload) as response:
                if response.status != 200:
                    print(f"Failed to login file. Status: {response.status}")
                    return None
                
            async with session.get(file_url, headers=headers) as response:
                if response.status != 200:
                    print(f"Failed to download file. Status: {response.status}")
                    return None
                
                # Get filename from URL or Content-Disposition header
                filename = file_url.split('/')[-1]
                content_disposition = response.headers.get('Content-Disposition')
                if content_disposition:
                    import re
                    match = re.search(r'filename="(.+?)"', content_disposition)
                    if match:
                        filename = match.group(1)
                
                print(f"Processing file: {filename}")
                
                # Save file
                file_path = self.download_dir / filename
                with open(file_path, 'wb') as f:
                    f.write(await response.read())
                
                file_info = {
                    "name": filename,
                    "url": str(file_path),
                    "type": self._get_file_type(filename)
                }
                print(f"File processed successfully: {file_info}")
                return file_info
                
        except Exception as e:
            print(f"Error downloading file {file_url}: {str(e)}")
            return None

    def _get_file_type(self, filename: str) -> str:
        """
        Determine the type of file based on its extension.
        """
        if not filename or not isinstance(filename, str):
            return 'other'
            
        parts = filename.lower().split('.')
        if len(parts) < 2:
            return 'other'
            
        ext = parts[-1]
        if ext in ['pdf']:
            return 'document'
        elif ext in ['doc', 'docx']:
            return 'word'
        elif ext in ['ppt', 'pptx']:
            return 'presentation'
        elif ext in ['txt']:
            return 'text'
        return 'other'

    def _get_base_domain(self, url: str) -> str:
        """
        Extract the base domain from a URL.
        Example: https://example.ch/blah/blah -> https://example.ch
        """
        try:
            parsed = urlparse(url)
            # Get the scheme (http/https) and netloc (domain)
            base = f"{parsed.scheme}://{parsed.netloc}"
            return base
        except Exception as e:
            print(f"Error parsing URL {url}: {str(e)}")
            return url

    def _get_file_type(self, filename: str) -> str:
        """
        Determine the type of file based on its extension.
        """
        if not filename or not isinstance(filename, str):
            return 'other'
            
        parts = filename.lower().split('.')
        if len(parts) < 2:
            return 'other'
            
        ext = parts[-1]
        if ext in ['pdf']:
            return 'document'
        elif ext in ['doc', 'docx']:
            return 'word'
        elif ext in ['ppt', 'pptx']:
            return 'presentation'
        elif ext in ['txt']:
            return 'text'
        return 'other' 
