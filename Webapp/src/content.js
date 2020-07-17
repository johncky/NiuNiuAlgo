class Dashboard extends React.Component{
    constructor(props){
        super(props);
        this.state = {cur_nav: 'content',
                    data: props.data,
                    cur_data: props.data.algos_data.combined,
                    cur_content: 'main',
                    };
        this.change_algo_handler = this.change_algo.bind(this);
        this.change_content_handler = this.change_content.bind(this);
        this.update_data_handelr = this.update_data.bind(this);
        this.update_interval = null;
    }

    update_data(new_data, add_algo=false){
        if (add_algo==true){
            this.setState({data: new_data, cur_data: new_data.algos_data.combined});
        }
        else {
            this.setState({data: new_data});
        }
    }

    componentDidMount() {
        this.update_interval = setInterval(() => get_summary(this.update_data_handelr), 1000*30);
      }

    componentWillUnmount() {
        clearInterval(this.update_interval);
    }

    change_algo(to_algo){
        this.setState({cur_data: this.state.data.algos_data[to_algo]});
    }

    change_content(to_content){
        if (to_content === 'settings'){
            this.setState({cur_content: 'settings'});
        }
        else if (to_content == 'main'){
            this.setState({cur_content: 'main'});
        }
        else{
            this.setState({cur_content: 'datails'});
        }
    }


    render(){
        if (this.state.cur_nav === 'content'){
            if (Object.keys(this.state.data.algos_data).length > 0){
                
                var navbar = {brandname: 'Algo.Py', navs: [
                    {active: true, text: 'Hook', is_dropdown:false, dropdown_content:[], on_click: ()=>{console.log('aaa');}, text_color:null},
                    {active: true, text: 'Algo', is_dropdown:true, dropdown_content:
                    Object.keys(this.state.data.algos_data).map((x)=> ({active: true, text: this.state.data.algos_data[x]['name'], is_dropdown:false,dropdown_content:[], on_click: ()=>{this.change_algo_handler(this.state.data.algos_data[x]['name'])}}))},
                ]};
                return(
                    <div>
                        <NavBar {...navbar} update_data_handelr={this.update_data_handelr}/>
                        <Content cur_content={this.state.cur_content} algos_data={this.state.cur_data} change_content_handler={this.change_content_handler}/>
                    </div>
                );



            }

            else{
                var navbar = {brandname: 'Algo.Py', navs: [
                    {active: true, text: 'Hook', is_dropdown:false, dropdown_content:[], on_click: ()=>{console.log('aaa');}, text_color:null},
                ]};
                return(
                    <div>
                        <NavBar {...navbar} update_data_handelr={this.update_data_handelr}/>
                        <NoAlgoPage/>
                    </div>
                )

            }

        }
    }
}


function NoAlgoPage(props){
    return (
        <h2>No Running Algo Found!</h2>
    )
}
function SubPageNav(props){
    return (
        <li className='nav-link active text-dark' key={props.to_content}>
        <a  className='nav-link active text-dark btn font-weight-bold'  onClick={props.handler}>{props.to_content}</a > 
        </li>
    );
}


function SubPageNavBar(props){
    return (
        <div className="container-fluid navbar navbar-expand navbar-dark flex-column flex-md-row bd-navbar bg-light" >
            <div className="collapse navbar-collapse" id="navbarSupportedContent">
                <a className="navbar-brand text-dark font-weight-bold" >{props.algo_name}</a>
                <ul className="navbar-nav mr-auto">
                    <SubPageNav to_content='Main' handler={() => props.handler('main')} />
                    <SubPageNav to_content='Settings' handler={() => props.handler('settings')} />
                    <SubPageNav to_content='Details' handler={() => props.handler('details')} />
                </ul>
            </div> 
        </div>
    );
}


class Content extends React.Component{
    constructor(props){
        super(props);
    }

    render(){
        if (this.props.cur_content === 'settings'){
            return (
                <main className="main" id="main">
                    <div className="container-fluid bg-light" >
                    <CardDeck>
                        <Card style={{ width: '100%'}} key="1" className="pblank-4 rounded shadow" bg='light' text='dark' border="light">
                            <Card.Body>
                                <SubPageNavBar algo_name = {this.props.algos_data.name} handler = {this.props.change_content_handler}/>
                                <Settings settings = {this.props.algos_data.settings} />
                            </Card.Body>
                        </Card>
                    </CardDeck>
                    </div>
                </main>
            );

        }
        else if (this.props.cur_content === "main") {
            return (
                <main className="main" id="main">
                    <div className="container-fluid bg-light" >
                    <CardDeck>
                        <Card style={{ width: '100%'}} key="1" className="pblank-4 rounded shadow" bg='light' text='dark' border="light">
                            <Card.Body>
                                <SubPageNavBar algo_name = {this.props.algos_data.name} handler = {this.props.change_content_handler}/>
                                <DashboardMain algos_data={this.props.algos_data}/>
                            </Card.Body>
                        </Card>
                    </CardDeck>
                    </div>
                </main>
            );
        }
        else if (this.props.cur_content === 'datails') {
            return (
                <main className="main" id="main">
                    <div className="container-fluid bg-light" >
                    <CardDeck>
                        <Card style={{ width: '100%'}} key="1" className="pblank-4 rounded shadow" bg='light' text='dark' border="light">
                            <Card.Body>
                                <SubPageNavBar algo_name = {this.props.algos_data.name} handler = {this.props.change_content_handler}/>
                                Details Content
                            </Card.Body>
                        </Card>
                    </CardDeck>
                    </div>
                </main>
            );
        }

    }

}


function SettingInputField(props){

    return (
            <InputGroup className="mb-2">
                <InputGroup.Prepend>
                <InputGroup.Text>{props.field}</InputGroup.Text>
                </InputGroup.Prepend>
                <FormControl placeholder={props.value}/>
                <Button variant="outline-secondary" onClick={()=>console.log('aa')}>Update</Button>
            </InputGroup>
    );

}

function Settings(props){
    var settings = props.settings;
    var headers = Object.keys(settings);
    headers = (headers.length >0)?headers : [];
    return (
        <CardDeck >
            <Card style={{ width: '100%'}} className="p-4 md-4 shadow" bg='light' text='dark' border="light">
                <Card.Body>
                    <Card.Title>Settings</Card.Title>
                    {headers.map(field => <SettingInputField field={field} value={settings[field]} key={field}/>)}
                </Card.Body>
            </Card>
        </CardDeck>
    );
}


class DashboardMain extends React.Component{
    constructor(props){
        super(props);
    }

    random_graph(){
    }

    componentDidMount(){
        this.interval = setInterval(() => this.random_graph.bind(this)(), 1000);
    }
    componentWillUnmount() {
        clearInterval(this.interval);
      }

    render(){

        return (
            <div>
            <CardDeck >
                <Card style={{ width: '30%'}} className="p-4 md-4 shadow" bg='light' text='dark' border="light">
                    <Card.Body>
                        <Card.Title>Performance</Card.Title>
                        <PerformanceCard algos_data={this.props.algos_data}/>
                    </Card.Body>
                </Card>

                <Card style={{ width: '30%' }} className="p-4 md-4 shadow" bg='light' text='dark' border="light">
                    <Card.Body>
                        <Card.Title>Status</Card.Title>
                        <StatsCard algos_data={this.props.algos_data}/>
                    </Card.Body>
                </Card>
                <Card style={{ width: '30%' }} className="p-4 md-4 shadow" bg='light' text='dark' border="light">
                    <Card.Body>
                        <Card.Title>Charts</Card.Title>
                        <ChartCard algos_data={this.props.algos_data}/>
                    </Card.Body>
                </Card>
            </CardDeck>
            <CardDeck>
                <Card style={{ width: '100%' }} className="pblank-1  mt-3 shadow" bg='light' text='dark' border="light">
                    <Card.Body>
                        <Card.Title>Positions</Card.Title>
                        <ScrollableTable headers={(this.props.algos_data.positions.length > 0)? Object.keys(this.props.algos_data.positions[0]) : []} data={this.props.algos_data.positions.map(x=> Object.values(x))}/>

                    </Card.Body>
                </Card>
            </CardDeck>
            <CardDeck>
                <Card style={{ width: '100%' }} className="pblank-1  mt-3 shadow" bg='light' text='dark' border="light">
                    <Card.Body>
                        <Card.Title>Pending Orders</Card.Title>
                        <ScrollableTable headers={(this.props.algos_data.pending.length > 0)? Object.keys(this.props.algos_data.pending[0]) : []} data={this.props.algos_data.pending.map(x=> Object.values(x))}/>
                    </Card.Body>
                </Card>
            </CardDeck>
            <CardDeck>
                <Card style={{ width: '100%' }} className="pblank-1  mt-3 shadow" bg='light' text='dark' border="light">
                    <Card.Body>
                        <Card.Title>Completed Orders</Card.Title>
                        <ScrollableTable headers={(this.props.algos_data.completed.length > 0)? Object.keys(this.props.algos_data.completed[0]) : []} data={this.props.algos_data.completed.map(x=> Object.values(x))}/>
                    </Card.Body>
                </Card>
            </CardDeck>
            </div>
        );
    }
}

function round_pct(pct, decimals){
    return Math.round(pct * (10**decimals)) / (10**decimals);
}

function PerformanceCard(props){
    var caret = (_return)=>(_return >= 0) ? (<svg className="bi bi-caret-up-fill" width="1em" height="1em" viewBox="0 2.5 16 16" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
    <path d="M7.247 4.86l-4.796 5.481c-.566.647-.106 1.659.753 1.659h9.592a1 1 0 0 0 .753-1.659l-4.796-5.48a1 1 0 0 0-1.506 0z"/>
  </svg>) : (<svg className="bi bi-caret-down-fill" width="1em" height="1em" viewBox="0 2.5 16 16" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
  <path d="M7.247 11.14L2.451 5.658C1.885 5.013 2.345 4 3.204 4h9.592a1 1 0 0 1 .753 1.659l-4.796 5.48a1 1 0 0 1-1.506 0z"/>
</svg>);

    var cal_return_data = (_return, _return_pct, is_benchmark, pv) => ((is_benchmark)? {return: round_pct(_return_pct * (pv - _return),2),
                                                                    text_color: (_return_pct >=0)? 'text-success font-weight-bold d-inline' : 'text-danger font-weight-bold d-inline',
                                                                    variant: (_return_pct >=0) ? 'success' : 'danger',
                                                                    return_str: '$'+Math.abs(round_pct((_return_pct * (pv - _return)),2).toString()) + '   ',
                                                                    return_pct_str: (round_pct(_return_pct * 100,2)).toString()+'%',
                                                                    sign: (_return_pct >=0) ? '+' : '-',
                                                                    caret: caret(_return_pct)}
                                                                    :
                                                                    {return: _return,
                                                                        text_color: (_return_pct >=0)? 'text-success font-weight-bold d-inline' : 'text-danger font-weight-bold d-inline',
                                                                        variant: (_return_pct >=0) ? 'success' : 'danger',
                                                                        return_str: '$'+Math.abs(round_pct(_return,2).toString()) + '   ',
                                                                        return_pct_str: (round_pct(_return_pct* 100, 2)).toString()+'%',
                                                                        sign: (_return_pct >=0) ? '+' : '-',
                                                                        caret: caret(_return_pct)})

    var daily = cal_return_data(props.algos_data.daily_return, props.algos_data.daily_return_pct, false, 0);
    var benchmark_daily = cal_return_data(props.algos_data.daily_return, props.algos_data.benchmark_daily_pct, true, props.algos_data.pv);
    var monthly = cal_return_data(props.algos_data.monthly_return, props.algos_data.monthly_return_pct, false, 0);
    var benchmark_monthly = cal_return_data(props.algos_data.monthly_return, props.algos_data.benchmark_monthly_pct, true, props.algos_data.pv);
    var total = cal_return_data(props.algos_data.net_pnl, props.algos_data.net_pnl_pct, false, 0);
    var total_benchmark = cal_return_data(props.algos_data.net_pnl, props.algos_data.benchmark_net_pnl_pct, true, props.algos_data.pv);

    return (
        <div>
            <hr/>
            <h6 > Daily PnL: </h6>
            <h5 className={daily.text_color}>
            {daily.sign}{daily.return_str}
            </h5>

            <h5 className='d-inline mh-100'>
            <Badge pill variant={daily.variant}>{daily.caret} {daily.return_pct_str}</Badge>
            </h5>

            <h3 className='d-inline font-weight-light'>   /  </h3>
            
            <h5 className={benchmark_daily.text_color}>
            {benchmark_daily.sign}{benchmark_daily.return_str}({props.algos_data.benchmark})
            </h5>

            <h5 className='d-inline mh-100'>
            <Badge pill variant={benchmark_daily.variant}>{benchmark_daily.caret} {benchmark_daily.return_pct_str}</Badge>
            </h5>

            <hr/>

            <h6 > Monthly PnL: </h6>
            <h5 className={monthly.text_color}>
            {monthly.sign}{monthly.return_str}
            </h5>

            <h5 className='d-inline mh-100'>
            <Badge pill variant={monthly.variant}>{monthly.caret} {monthly.return_pct_str}</Badge>
            </h5>

            <h3 className='d-inline font-weight-light'>   /  </h3>
            
            <h5 className={benchmark_monthly.text_color}>
            {benchmark_monthly.sign}{benchmark_monthly.return_str}({props.algos_data.benchmark})
            </h5>

            <h5 className='d-inline mh-100'>
            <Badge pill variant={benchmark_monthly.variant}>{benchmark_monthly.caret} {benchmark_monthly.return_pct_str}</Badge>
            </h5>

            <hr/>

            <h6 > Total PnL: </h6>
            <h5 className={total.text_color}>
            {total.sign}{total.return_str}
            </h5>

            <h5 className='d-inline mh-100'>
            <Badge pill variant={total.variant}>{total.caret} {total.return_pct_str}</Badge>
            </h5>

            <h3 className='d-inline font-weight-light'>   /  </h3>
            
            <h5 className={total_benchmark.text_color}>
            {total_benchmark.sign}{total_benchmark.return_str}({props.algos_data.benchmark})
            </h5>

            <h5 className='d-inline mh-100'>
            <Badge pill variant={total_benchmark.variant}>{total_benchmark.caret} {total_benchmark.return_pct_str}</Badge>
            </h5>

            <hr/>

            <Table hover>
            <thead>
                <tr>
                <th>Range</th>
                <th>Actual</th>
                <th>Benchmark</th>
                <th>Outperformance</th>
                </tr>
            </thead>
            <tbody>

                <tr>
                <td>Annualized Return: </td>
                <td>{(round_pct(props.algos_data.annualized_return* 100,2)).toString() + '% '}</td>
                <td>{(round_pct(props.algos_data.benchmark_annualized_return* 100, 2)).toString() + '% '}</td>
                <td>{round_pct((props.algos_data.annualized_return - props.algos_data.benchmark_annualized_return)*100,2) + '%'}</td>
                </tr>

                <tr>
                <td>Daily: </td>
                <td>{daily.return_pct_str}</td>
                <td>{benchmark_daily.return_pct_str}</td>
                <td>{round_pct((props.algos_data.daily_return_pct - props.algos_data.benchmark_daily_pct)*100,2) + '%'}</td>
                </tr>

                <tr>
                <td>Monthly: </td>
                <td>{monthly.return_pct_str}</td>
                <td>{benchmark_monthly.return_pct_str}</td>
                <td>{round_pct((props.algos_data.monthly_return_pct - props.algos_data.benchmark_monthly_pct)*100,2) + '%'}</td>
                </tr>


            </tbody>
            </Table>

        </div>
    );
};

function StatsCard(props){
    var button = (props.algos_data.status.toLowerCase() == 'running') ? (<svg className="bi bi-x-circle" width="1em" height="1em" viewBox="0 0 16 16" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
    <path fillRule="evenodd" d="M8 15A7 7 0 1 0 8 1a7 7 0 0 0 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16z"/>
    <path fillRule="evenodd" d="M11.854 4.146a.5.5 0 0 1 0 .708l-7 7a.5.5 0 0 1-.708-.708l7-7a.5.5 0 0 1 .708 0z"/>
    <path fillRule="evenodd" d="M4.146 4.146a.5.5 0 0 0 0 .708l7 7a.5.5 0 0 0 .708-.708l-7-7a.5.5 0 0 0-.708 0z"/>
  </svg>): (<svg className="bi bi-play" width="1em" height="1em" viewBox="0 0 16 16" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
  <path fillRule="evenodd" d="M10.804 8L5 4.633v6.734L10.804 8zm.792-.696a.802.802 0 0 1 0 1.392l-6.363 3.692C4.713 12.69 4 12.345 4 11.692V4.308c0-.653.713-.998 1.233-.696l6.363 3.692z"/>
</svg>);
    return(
        <div>
            <hr/>
            <h6 className='d-inline'>
                {'Algo Status:  '}
            </h6>

            <h6 className='d-inline'>
            {props.algos_data.status + '   '}
            <a className='navbar-brand mr-0 mr-md-0 text-dark' href='/'>
            {button}
            </a>
            </h6>

            <hr/>

            <Table hover>
            <thead>
                <tr>
                <th>Portfolio Value:</th>
                <th>{'$'+props.algos_data.pv}</th>
                <th>%</th>
                </tr>
            </thead>
            <tbody>

                <tr>
                <td>Assets: </td>
                <td>{'$'+(props.algos_data.pv - props.algos_data.cash)}</td>
                <td>{round_pct(((props.algos_data.pv - props.algos_data.cash) / props.algos_data.pv * 100),2) + '%'}</td>
                </tr>

                <tr>
                <td>Cash: </td>
                <td>{'$'+props.algos_data.cash}</td>
                <td>{round_pct((props.algos_data.cash / props.algos_data.pv * 100),2) + '%'}</td>
                </tr>

                <tr>
                <td>Margin: </td>
                <td>{'$'+props.algos_data.margin}</td>
                <td>{round_pct((props.algos_data.margin / props.algos_data.pv * 100),2) + '%'}</td>
                </tr>


            </tbody>
            </Table>
            <hr/>

            <Table hover>
            <thead>
                <tr>
                <th>Metrics &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; </th>
                <th>{props.algos_data.name}&nbsp;&nbsp;  </th>
                <th>{props.algos_data.benchmark}</th>
                </tr>
            </thead>
            <tbody>

                <tr>
                <td>Sharpe:</td>
                <td>{round_pct(props.algos_data.sharpe, 2)}</td>
                <td>{round_pct(props.algos_data.benchmark_sharpe, 2)}</td>
                </tr>

                <tr>
                <td>Beta: </td>
                <td>{round_pct(props.algos_data.beta, 2)}</td>
                <td>{1}</td>
                </tr>

                <tr>
                <td>Sortino: </td>
                <td>In progress</td>
                <td>In Progress</td>
                </tr>

                <tr>
                <td>Txn Cost:  </td>
                <td>{props.algos_data.txn_cost_total}</td>
                </tr>

            </tbody>
            </Table>

        </div>
    );
}

function ChartCard(props){
    return (
        <div>
            <Tabs defaultActiveKey="PV" id="uncontrolled-tab-example" className='myClass'>
                <Tab eventKey="PV" title="PV">
                    <SplineAreaChart title="Portfolio Value" data={props.algos_data.PV}/>
                </Tab>

                <Tab eventKey="EV" title="EV">
                    <SplineAreaChart title="Equity Value" data={props.algos_data.EV}/>
                </Tab>

                <Tab eventKey="Cash" title="Cash">
                    <SplineAreaChart title="Cash" data={props.algos_data.Cash}/>
                </Tab>

                <Tab eventKey="Margin" title="Margin">
                    <SplineAreaChart title="Margin" data={props.algos_data.Margin}/>
                </Tab>
            </Tabs>
        </div>
    );
}

function Postions(props){
    return(
        <div>

        </div>
    );
}


ReactDOM.render(<Dashboard data={data}/>, document.getElementById("main_holder"));

