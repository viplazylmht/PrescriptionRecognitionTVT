const fileInput = document.getElementById('upload');
const mess = document.getElementById('message');
const mess_area = document.getElementById('mess_area');

let auto_reload;

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

                let s = '<p class="text-white">' + result.result[0] + "</p>";

                for (let i = 1; i < 4; i++) {
                    line = result.result[i][i - 1]
                    s += '<p class="text-white">' + i + ". " + line[0] + "</p>";
                }

                mess_area.innerHTML = s;
            }
            else if (result.status === 'queued') {
                mess.textContent = 'Job quequed';
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