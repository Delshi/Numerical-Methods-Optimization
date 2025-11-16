import os
from pathlib import Path

def get_file_structure(directory, ignore_dirs=None, ignore_extensions=None, max_depth=None, current_depth=0):
    """
    Рекурсивно собирает файловую структуру директории.
    
    :param directory: Путь к директории
    :param ignore_dirs: Список директорий для игнорирования
    :param ignore_extensions: Список расширений файлов для игнорирования
    :param max_depth: Максимальная глубина рекурсии
    :param current_depth: Текущая глубина рекурсии (для внутреннего использования)
    :return: Строка с файловой структурой
    """
    if ignore_dirs is None:
        ignore_dirs = ['.git', '__pycache__', 'venv', 'node_modules']
    if ignore_extensions is None:
        ignore_extensions = ['.pyc', '.tmp', '.cache']
    
    structure = []
    indent = '    ' * current_depth
    
    try:
        with os.scandir(directory) as entries:
            for entry in sorted(entries, key=lambda e: e.name):
                if entry.name.startswith('.'):
                    continue
                
                if entry.is_dir() and entry.name not in ignore_dirs:
                    structure.append(f"{indent}📁 {entry.name}/")
                    if max_depth is None or current_depth < max_depth:
                        try:
                            sub_structure = get_file_structure(
                                entry.path, 
                                ignore_dirs, 
                                ignore_extensions, 
                                max_depth, 
                                current_depth + 1
                            )
                            structure.append(sub_structure)
                        except PermissionError:
                            structure.append(f"{indent}    [Permission denied]")
                
                elif entry.is_file():
                    if any(entry.name.endswith(ext) for ext in ignore_extensions):
                        continue
                    file_size = os.path.getsize(entry.path)
                    structure.append(f"{indent}📄 {entry.name} ({file_size} bytes)")
    except PermissionError:
        return f"{indent}[Permission denied to access {directory}]"
    
    return '\n'.join(structure)

def save_structure_to_file(structure, output_file='file_structure.txt'):
    """Сохраняет файловую структуру в текстовый файл."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(structure)
    print(f"Файловая структура сохранена в {output_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Генератор файловой структуры для промптов')
    parser.add_argument('directory', type=str, nargs='?', default='.', 
                       help='Директория для сканирования (по умолчанию: текущая)')
    parser.add_argument('--output', type=str, default='file_structure.txt',
                       help='Имя выходного файла (по умолчанию: file_structure.txt)')
    parser.add_argument('--max-depth', type=int, default=None,
                       help='Максимальная глубина рекурсии')
    parser.add_argument('--ignore-dirs', type=str, nargs='*', 
                       default=['.git', '__pycache__', 'venv', 'node_modules'],
                       help='Директории для игнорирования')
    parser.add_argument('--ignore-ext', type=str, nargs='*',
                       default=['.pyc', '.tmp', '.cache'],
                       help='Расширения файлов для игнорирования')
    
    args = parser.parse_args()
    
    print(f"Сканирование директории: {args.directory}")
    structure = get_file_structure(
        args.directory,
        ignore_dirs=args.ignore_dirs,
        ignore_extensions=args.ignore_ext,
        max_depth=args.max_depth
    )
    
    # Добавляем заголовок с информацией о директории
    full_path = os.path.abspath(args.directory)
    header = f"Файловая структура директории: {full_path}\n"
    header += f"Глубина сканирования: {'без ограничений' if args.max_depth is None else args.max_depth}\n"
    header += f"Игнорируемые директории: {', '.join(args.ignore_dirs)}\n"
    header += f"Игнорируемые расширения: {', '.join(args.ignore_ext)}\n\n"
    
    full_structure = header + structure
    print("\n" + full_structure)
    
    save_structure_to_file(full_structure, args.output)

if __name__ == "__main__":
    main()
