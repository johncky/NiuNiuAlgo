tmp = to_algo=='combined' ? Object.assign({},this.state.data.combined) : Object.assign({},this.state.data.algos_data[to_algo]);


var navbar = {brandname: 'Algo.Py', navs: [
    {active: true, text: 'Hook', is_dropdown:false, dropdown_content:[], on_click: ()=>{console.log('aaa');}, text_color:null},
    {active: true, text: 'Algo', is_dropdown:true, dropdown_content:[
        {active: true, text: 'combined', is_dropdown:false, dropdown_content:[], on_click: ()=>{this.change_algo_handler('combined')}, text_color:null}
    ].concat(Object.keys(this.state.data.algos_data).map((x)=> ({active: true, text: this.state.data.algos_data[x]['name'], is_dropdown:false,dropdown_content:[], on_click: ()=>{this.change_algo_handler(data.algos_data[x].name)}})))},
]};