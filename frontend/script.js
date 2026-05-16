document.addEventListener('DOMContentLoaded', () => {
    const fileElem = document.getElementById('fileElem');
    const uploadBtn = document.querySelector('.upload-btn');
    const resultContainer = document.getElementById('result-container');
    const origImg = document.getElementById('orig-img');
    const predImg = document.getElementById('pred-img');
    
    // Simulate initial panel open
    setTimeout(() => {
        document.querySelector('.left-panel').style.transform = 'translateX(0)';
        setTimeout(() => {
            document.querySelector('.left-panel').style.transform = '';
        }, 3000);
    }, 1000);

    fileElem.addEventListener('change', handleFiles, false);

    function handleFiles(e) {
        const files = this.files;
        if (files.length) {
            uploadFile(files[0]);
        }
    }

    function uploadFile(file) {
        const url = '/segment';
        const formData = new FormData();
        formData.append('file', file);

        uploadBtn.innerText = 'TRANSMITTING...';
        uploadBtn.style.background = 'var(--amber)';
        uploadBtn.style.color = '#000';
        uploadBtn.style.borderColor = 'var(--amber)';
        uploadBtn.style.boxShadow = '0 0 15px var(--amber)';

        fetch(url, {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            uploadBtn.innerText = 'TRANSMIT IMAGE';
            uploadBtn.style = '';

            resultContainer.classList.remove('hidden');
            origImg.src = data.original_image;
            predImg.src = data.segmented_image;

            // Open right panel to show results
            document.querySelector('.right-panel').style.transform = 'translateX(0)';
        })
        .catch(() => {
            uploadBtn.innerText = 'TRANSMISSION FAILED';
            uploadBtn.style.background = 'var(--red)';
            uploadBtn.style.color = '#fff';
            uploadBtn.style.borderColor = 'var(--red)';
            uploadBtn.style.boxShadow = '0 0 15px var(--red)';
            
            setTimeout(() => {
                uploadBtn.innerText = 'TRANSMIT IMAGE';
                uploadBtn.style = '';
            }, 3000);
        });
    }
});
