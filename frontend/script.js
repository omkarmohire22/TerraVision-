const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('fileElem');
const uploadForm = document.getElementById('upload-form');
const loader = document.getElementById('loader');
const resultsArea = document.getElementById('results-area');
const origImg = document.getElementById('orig-img');
const predImg = document.getElementById('pred-img');

// Prevent default drag behaviors
;['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false)
    document.body.addEventListener(eventName, preventDefaults, false)
})

function preventDefaults(e) {
    e.preventDefault()
    e.stopPropagation()
}

// Highlight drop area when item is dragged over it
;['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, highlight, false)
})

    ;['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false)
    })

function highlight(e) {
    dropArea.classList.add('highlight')
}

function unhighlight(e) {
    dropArea.classList.remove('highlight')
}

// Handle dropped files
dropArea.addEventListener('drop', handleDrop, false)

function handleDrop(e) {
    let dt = e.dataTransfer
    let files = dt.files
    handleFiles(files)
}

// Handle file input click
fileInput.addEventListener('change', function () {
    handleFiles(this.files);
});

function handleFiles(files) {
    if (files.length > 0) {
        uploadFile(files[0]);
    }
}

function uploadFile(file) {
    // Hide upload form and show loader
    uploadForm.classList.add('hidden');
    loader.classList.remove('hidden');
    resultsArea.classList.add('hidden');

    let url = '/segment';
    let formData = new FormData();
    formData.append('file', file);

    fetch(url, {
        method: 'POST',
        body: formData
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                alert('Error: ' + data.error);
                resetUI();
                return;
            }

            // Update Images
            origImg.src = data.original_image;
            predImg.src = data.segmented_image;

            // Hide loader and show results & form
            loader.classList.add('hidden');
            uploadForm.classList.remove('hidden');
            resultsArea.classList.remove('hidden');
        })
        .catch(error => {
            console.error('Error:', error);
            alert('There was an error processing your image.');
            resetUI();
        });
}

function resetUI() {
    loader.classList.add('hidden');
    uploadForm.classList.remove('hidden');
}
