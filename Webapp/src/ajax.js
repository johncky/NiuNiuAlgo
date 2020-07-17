
function get_summary(update_handler, add_algo=false){
    fetch('/data')
        .then(function(response) {
            return response.json();
        })
            .then(function(new_data) {
                update_handler(new_data, add_algo);
            })
                .catch((err)=> {
                    console.log(err);
                })

}


function add_algo(update_handler){
    var algo_ip = document.getElementById("mod_algo_ip").value;
    fetch('/add_algo?'+'ip='+algo_ip)
        .then(function (response){
            return  response.json();
        }
        )
            .then(function (result){
                alert(result.response);
                get_summary(update_handler, true);
            })
            .catch(()=> {
                alert('Failed to add algo');
            })

}

function remove_algo(update_handler){
    var algo_ip = document.getElementById("mod_algo_ip").value;
    fetch('/remove_algo?'+'ip='+algo_ip)
        .then(function (response){
            return  response.json();
        }
        )
            .then(function (result){
                alert(result.response);
                get_summary(update_handler);
            })
            .catch(()=> {
                alert('Failed to remove algo');
            })

}