const fileInput = document.getElementById('upload');
const mess = document.getElementById('message');
const mess_area = document.getElementById('mess_area');

let auto_reload;

function getFirstItem(arr) {
    return arr[0]
}

const reload_result = async (job_id) => {
    const res = await fetch("/api/v1/results/" + job_id,
        {
            method: 'GET',
        }).then(response => {
            return response.json();
        }).then(result => {
            console.log(result);

            if (result.status === 'completed') {
                clearInterval(auto_reload);

                mess.textContent = 'Job completed!';

                let s = '<p class="text-white">' + result.result[0] + "</br>";
                haha = result;
                s += haha.result.slice(1).map(getFirstItem).map(JSON.stringify).join("</br></br>").replaceAll("\\n", "</br>")
                s += "</p>";

                mess_area.innerHTML = s;
            }
            else if (result.status === 'queued') {
                mess.textContent = 'Job quequed';
            }
            else if (result.status === 'handle_output') {
                mess.textContent = result.message;
            }
            else {
                mess.textContent = 'Job is processing. Please wait...';
            }

            return result;
        });

    return res;
};
let haha;
const caption_me = async () => {
    const file = fileInput.files[0];

    // handle file not found
  
    const data = new FormData();
    data.append('file', file);
    mess.textContent = 'Uploading...';
    mess_area.innerHTML = "";

    const processedImage = await fetch("/api/v1/captionme",
        {
            method: 'POST',
            body: data
        }).then(response => {
            return response.json();
        }).then(result => {
            console.log(result);
            haha = result;
            if (result.error) {
                mess.textContent = "Error: " + result.error;

                return result;
            }

            const job_id = result.result.job_id;

            mess.textContent = 'Job ID created: ' + job_id;

            auto_reload = setInterval(() => { reload_result(job_id); }, 1000);

            return result;
        });
};