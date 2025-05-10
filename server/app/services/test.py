import aiohttp
import asyncio

async def download_pdf():
    url = "https://las.inf.ethz.ch/courses/pai-f24/slides/pai-01-introduction.pdf"
    auth = aiohttp.BasicAuth("ml", "predict")

    async with aiohttp.ClientSession(auth=auth) as session:
        async with session.get(url) as response:
            print(f"Status: {response.status}")
            if response.status == 200:
                content = await response.read()
                with open("file.pdf", "wb") as f:
                    f.write(content)
            else:
                print("Failed to download file")

asyncio.run(download_pdf())
