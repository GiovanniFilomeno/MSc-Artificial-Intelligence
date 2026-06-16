import asyncio
import aiofiles
import os

async def read_file_async(file_path: str) -> list[str]:
    lines = []
    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
        async for line in f:
            lines.append(line.rstrip('\n'))
    return lines

async def merge_files_concurrently(input_files: list[str], output_file: str):
    tasks = [asyncio.create_task(read_file_async(f)) for f in input_files]
    
    results = await asyncio.gather(*tasks)
    
    all_lines = []
    for lines in results:
        all_lines.extend(lines)
    
    all_lines.sort()
    
    async with aiofiles.open(output_file, 'w', encoding='utf-8') as out_f:
        for line in all_lines:
            await out_f.write(line + '\n')


if __name__ == "__main__":
    # Copy-Paste from exercise
    async def main():
        input_files = ["file1.txt", "file2.txt", "file3.txt"]
        output_file = "merged.txt"
        
        for i, f in enumerate(input_files, start=1):
            if not os.path.exists(f):
                with open(f, 'w', encoding='utf-8') as ff:
                    ff.write(f"Line {i}_1\nLine {i}_2\nLine {i}_3\n")
        
        await merge_files_concurrently(input_files, output_file)
        print(f"Merged {input_files} into {output_file}.")

    asyncio.run(main())
