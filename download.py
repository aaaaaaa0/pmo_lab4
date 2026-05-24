import os
import urllib.request

def main():
    os.makedirs('data', exist_ok=True)
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    dest = 'data/titanic.csv'
    print(f'Downloading {url} ...')
    urllib.request.urlretrieve(url, dest)
    print(f'Saved to {dest}')

if __name__ == '__main__':
    main()